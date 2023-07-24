import argparse
import os
import shutil
import time
import hashlib

import numpy as np
import pandas as pd

from typing import List
from dask_sql import Context

from src.default_paths import path_omop, path_root
from src.utils import list_dir

LABELERS: List[str] = [
    "mortality",
    "long_los",
    "readmission",
    # Lab-based tasks
    "thrombocytopenia_lab",
    "hyperkalemia_lab",
    "hypoglycemia_lab",
    "hyponatremia_lab",
    "anemia_lab",
]


def register_omop_tables(c: Context, path_omop: str, file_format: str = "parquet"):
    """
    Register tables in `path_omop` directory into Context so that they are query-able.
    Assumes each table is a subdirectory containing fileparts with format
    specified by `file_format`.

    Args:
        c: dask_sql.Context.
        path_omop: path to directory containing omop tables.
        file_format: format of fileparts. Defaults to parquet.

    Returns:
        dask_sql.Context

    Raises:
        N/A
    """
    for table in list_dir(path_omop):
        c.create_table(table, os.path.join(path_omop, table, f"*.{file_format}"))
    return c


def generate_id_hash(row, columns: list, max_digit: int = 15):
    s = "-".join([str(row[x]) for x in columns])

    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 10**max_digit


# Custom date functions for dask_sql
def date_diff_years(a, b):
    return (a - b).dt.days / 365.25


def date_diff_days(a, b):
    return (a - b).dt.days


def convert_to_date(a):
    return a.dt.date


# Queries
def create_ip_cohort(c):
    """
    Creates an inpatient cohort defined as visits with `visit_concept_id` IN (9201, 262).
    Combines overlapping / contiguous inpatient visits.
    If patient has multiple inpatient visits, then 1 is pseudorandomly selected. The selection
    process is deterministic.

    Index datetime is set as midnight (23:59:00) of admission date. Visits in which discharge
    or death occurred before index datetime are removed.

    No neonates.

    Args:
        c: dask_sql.Context.

    Returns:
        dask_sql.Context

    Raises:
        N/A
    """

    # get IP visits (9201, 262)
    visits = c.sql(
        """
    SELECT
        p.person_id,
        visit_start_datetime as admit_datetime,
        visit_end_datetime as discharge_datetime
    FROM visit_occurrence v
    INNER JOIN person p
        ON v.person_id = p.person_id
    WHERE
        visit_concept_id IN (9201, 262)
        AND visit_end_datetime is not NULL
        AND visit_start_datetime is not NULL
    """
    ).compute()

    # use dask to get lag discharge time
    visits_with_lag = visits.assign(
        lag_discharge_datetime=visits["discharge_datetime"]
    ).sort_values(["person_id", "admit_datetime", "discharge_datetime"])
    visits_with_lag["lag_discharge_datetime"] = (
        visits_with_lag["lag_discharge_datetime"]
        .shift(1)
        .where(visits_with_lag["person_id"].eq(visits_with_lag["person_id"].shift(1)))
    )

    c.create_table("ip_visits_with_lag", visits_with_lag)

    # use dask sql to create initial cohort
    cohort = c.sql(
        """
    WITH visits_with_group_boundaries AS (
        SELECT v.*
            ,CASE
                WHEN (
                    admit_datetime = lag_discharge_datetime
                    OR admit_datetime < lag_discharge_datetime
                )
                then 0
                else 1
            END AS start_new_group
        FROM ip_visits_with_lag v
    ),
    visits_with_groups AS (
        SELECT v.*
            ,SUM(start_new_group) OVER (
                PARTITION BY person_id
                ORDER BY admit_datetime, discharge_datetime
            ) as group_id
        FROM visits_with_group_boundaries v
    ),
    visits_rolledup AS (
        SELECT
            ROW_NUMBER() OVER (ORDER BY person_id) AS person_row_id
            ,person_id
            ,MIN(admit_datetime) as admit_datetime
            ,MAX(discharge_datetime) as discharge_datetime
        FROM visits_with_groups
        GROUP BY person_id, group_id
    )
    SELECT c.*
        ,date_diff_days(admit_datetime, birth_datetime) AS age_at_admission_days
        ,CASE WHEN gender_concept_id = 8507 THEN 'M' WHEN gender_concept_id = 8532 THEN 'F' ELSE 'OTHER' END AS sex
    FROM visits_rolledup c
    LEFT JOIN death d ON c.person_id = d.person_id
    LEFT JOIN person p ON c.person_id = p.person_id

    WHERE
        discharge_datetime > admit_datetime
        AND

        -- remove deaths that occurred prior to midnight of admission (used as index datetime, see below)
        (
            convert_to_date(d.death_date) > convert_to_date(admit_datetime)
            OR d.death_date IS NULL
        )
    """
    ).compute()

    # 28 days or older (remove neonates)
    cohort = cohort.query("age_at_admission_days>=28")

    # assign midnight on admission date as the index datetime
    cohort = cohort.assign(
        index_datetime=(
            pd.to_datetime(pd.to_datetime(cohort["admit_datetime"]).dt.date)
            + pd.Timedelta(hours=23, minutes=59)
        )
    )

    # remove discharges before index datetime
    cohort = cohort.query("discharge_datetime > index_datetime")

    # hash row_id for deterministic selection of admission
    cohort = cohort.assign(
        person_row_id=cohort.apply(
            lambda row: generate_id_hash(
                columns=["person_id", "admit_datetime", "discharge_datetime"],
                max_digit=15,
                row=row,
            ),
            axis=1,
        )
    )

    cohort = (
        cohort.sort_values(["person_id", "person_row_id"])
        .groupby("person_id")
        .first()
        .reset_index()[
            [
                "person_id",
                "index_datetime",
                "admit_datetime",
                "discharge_datetime",
                "age_at_admission_days",
                "sex",
            ]
        ]
    )

    return cohort


"""
Operational Labelers

Args:
    c: dask_sql.Context that contains OMOP tables and the inpatient cohort table

Returns:
    pandas df with the following columns:
        patient_id: person_id
        prediction_time: index_datetime
        label_type: "boolean"
        value: boolean label of whether label was observed between prediction time and discharge
        death_date: date of death
"""


def labeler_mortality(c):
    """
    Inpatient mortality defined as patient death occurring during the admission

    """

    df = c.sql(
        """
    SELECT c.*
        ,d.death_date
        ,CASE
            WHEN convert_to_date(d.death_date) > convert_to_date(index_datetime)
                AND convert_to_date(d.death_date) <= convert_to_date(discharge_datetime)
            THEN 1
            ELSE 0
            END as mortality
    FROM cohort c
    LEFT JOIN death d
        ON c.person_id = d.person_id
    """
    ).compute()

    df = (
        df.assign(label_type="boolean")
        .assign(value=df["mortality"] == 1)
        .rename(
            columns={
                "person_id": "patient_id",
                "index_datetime": "prediction_time",
            }
        )[["patient_id", "prediction_time", "label_type", "value", "death_date"]]
    )

    return df


def labeler_los(c):
    """
    Inpatient long length of stay defined as stays >= 7 days

    """

    df = c.sql(
        """
    SELECT c.*
        ,date_diff_days(discharge_datetime, admit_datetime) AS los_days
        ,CASE
            WHEN date_diff_days(discharge_datetime, admit_datetime) >= 7
            THEN 1
            ELSE 0
            END as long_los
    FROM cohort c
    """
    ).compute()

    df = (
        df.assign(label_type="boolean")
        .assign(value=df["long_los"] == 1)
        .rename(
            columns={
                "person_id": "patient_id",
                "index_datetime": "prediction_time",
            }
        )[
            [
                "patient_id",
                "prediction_time",
                "label_type",
                "value",
                "admit_datetime",
                "discharge_datetime",
                "los_days",
            ]
        ]
    )

    return df


def labeler_readmission(c):
    """
    30-day re-admission defined as whether a readmission occurred in 30 days or less.

    Here, index_datetime is set as midnight on discharge day.
    """

    df = c.sql(
        """
    WITH all_visits AS (
        SELECT
            p.person_id,
            visit_start_datetime as admit_datetime,
            visit_end_datetime as discharge_datetime
        FROM visit_occurrence v
        INNER JOIN person p
            ON v.person_id = p.person_id
        WHERE
            visit_concept_id IN (9201, 262)
            AND visit_end_datetime is not NULL
            AND visit_start_datetime is not NULL
    ),
    cohort_with_future_visits AS (
        SELECT
            c.*
            ,v.admit_datetime AS future_admit_datetime
            ,date_diff_days(v.admit_datetime, c.discharge_datetime) AS days_to_future_visit
        FROM cohort c
        LEFT JOIN all_visits v ON c.person_id = v.person_id
            AND v.admit_datetime > c.discharge_datetime
    ),
    cohort_with_future_visits_ranked AS (
        SELECT
            c.*
            ,ROW_NUMBER() OVER(
                PARTITION BY person_id ORDER BY days_to_future_visit
                ) AS visit_rank
        FROM cohort_with_future_visits c
    )
    SELECT c.person_id
        ,c.admit_datetime
        ,c.discharge_datetime
        ,v.future_admit_datetime as readmission_datetime
        ,v.days_to_future_visit as readmission_days
        ,CASE
            WHEN v.days_to_future_visit <=30
            THEN 1
            ELSE 0
            END AS readmission
    FROM cohort c
    LEFT JOIN (
            SELECT *
            FROM cohort_with_future_visits_ranked
            WHERE visit_rank=1
        ) v
        ON c.person_id = v.person_id
    WHERE
        convert_to_date(future_admit_datetime) > convert_to_date(c.discharge_datetime)
        OR future_admit_datetime IS NULL
    """
    ).compute()

    df = df.assign(
        index_datetime=(
            pd.to_datetime(pd.to_datetime(df["discharge_datetime"]).dt.date)
            + pd.Timedelta(hours=23, minutes=59)
        )
    )

    df = (
        df.assign(label_type="boolean")
        .assign(value=df["readmission"] == 1)
        .rename(
            columns={
                "person_id": "patient_id",
                "index_datetime": "prediction_time",
            }
        )[
            [
                "patient_id",
                "prediction_time",
                "label_type",
                "value",
                "discharge_datetime",
                "readmission_datetime",
                "readmission_days",
            ]
        ]
    )

    return df


"""
Single-threshold based lab labelers
"""


def get_lab_labels(
    c: Context,
    concept_ids: list,
    unit_concept_ids: dict,
    min_max: str,
    threshold: str,
):
    """
    Get lab labels using set of concept IDs, unit conversion rules, and a single threshold value

    Args:
        c: dask_sql.Context,
        concept_ids: list of concept IDs to search (will include all descendants)
        unit_concept_ids: dictionary with unit_concept_id as keys, and conversion rule as values
        min_max: "min" / "max"
        threshold: lower or upper limit
    """

    if min_max == "min":
        sign = "<"
        order = "ASC"
    elif min_max == "max":
        sign = ">"
        order = "DESC"
    else:
        raise ValueError("min_max must be min or max")

    sql_concept_ids = ",".join([str(x) for x in concept_ids])
    sql_unit_concept_ids = ",".join([str(k) for k in unit_concept_ids.keys()])
    sql_unit_conversion = " ".join(
        [
            f"WHEN unit_concept_id = {str(k)} THEN value_as_number {str(v)}"
            for k, v in unit_concept_ids.items()
        ]
    )

    query = f"""
    WITH concepts AS
    (
        SELECT
            c.concept_id, c.concept_name
        FROM concept c
        WHERE c.concept_id in ({sql_concept_ids})

        UNION DISTINCT

        SELECT
            c.concept_id, c.concept_name
        FROM concept c
        INNER JOIN concept_ancestor ca
            ON c.concept_id = ca.descendant_concept_id
            AND ca.ancestor_concept_id in ({sql_concept_ids})
            AND c.invalid_reason is null
    ),
    all_measurements AS
    (
        SELECT
            cohort.*
            ,CASE
                {sql_unit_conversion}
            ELSE NULL
            END AS value_as_number
            ,measurement_datetime
        FROM cohort
        INNER JOIN measurement m
            ON cohort.person_id = m.person_id
            AND m.measurement_datetime >= cohort.admit_datetime
            AND m.measurement_datetime <= cohort.discharge_datetime
            AND m.unit_concept_id IN ({sql_unit_concept_ids})
            AND m.value_as_number <> 9999999
        INNER JOIN concepts c
            ON m.measurement_concept_id = c.concept_id
    ),
    measurements_ordered AS
    (
        SELECT m.*
            ,ROW_NUMBER() OVER(
                PARTITION BY person_id, admit_datetime, discharge_datetime
                ORDER BY value_as_number {order}
                ) rn
        FROM all_measurements m
        WHERE value_as_number IS NOT NULL
    ),
    selected_measurements AS
    (
        SELECT
            person_id
            ,index_datetime
            ,discharge_datetime
            ,value_as_number
            ,measurement_datetime
        FROM measurements_ordered
        WHERE rn = 1
    )
    SELECT c.*
        ,m.value_as_number AS min_max_value
        ,m.measurement_datetime
        ,CASE WHEN m.value_as_number {sign} {threshold} THEN 1 ELSE 0 END AS label
    FROM cohort c
    LEFT JOIN selected_measurements m
        ON c.person_id = m.person_id
        AND c.index_datetime = m.index_datetime
        AND c.discharge_datetime = m.discharge_datetime
    """
    print(query)
    df = c.sql(query).compute()

    df = (
        df.assign(label_type="boolean")
        .assign(value=df["label"] == 1)
        .assign(threshold=threshold)
        .rename(
            columns={
                "person_id": "patient_id",
                "index_datetime": "prediction_time",
            }
        )[
            [
                "patient_id",
                "prediction_time",
                "label_type",
                "value",
                "min_max_value",
                "measurement_datetime",
                "threshold",
            ]
        ]
    )

    # exclude onsets that occurred prior to prediction_time
    df = df.query(
        "\
    (value==True and measurement_datetime > prediction_time) \
    or value==False\
    "
    )

    return df


def labeler_thrombocytopenia(c):
    concept_ids: list = [37037425, 40654106]
    unit_concept_ids = {
        8848: "",
        8961: "",
        9444: "",
    }
    min_max = "min"
    threshold = "50"

    return get_lab_labels(c, concept_ids, unit_concept_ids, min_max, threshold)


def labeler_hyperkalemia(c):
    concept_ids: list = [40653595, 37074594, 40653596]
    unit_concept_ids = {
        8753: "",
        8840: "/18",
        9557: "",
    }
    min_max = "max"
    threshold = "7"

    return get_lab_labels(c, concept_ids, unit_concept_ids, min_max, threshold)


def labeler_hyponatremia(c):
    concept_ids: list = [40653762]
    unit_concept_ids = {
        8753: "",
        9557: "",
    }
    min_max = "min"
    threshold = "125"

    return get_lab_labels(c, concept_ids, unit_concept_ids, min_max, threshold)


def labeler_hypoglycemia(c):
    concept_ids: list = [4144235, 1002597]
    unit_concept_ids = {
        8753: "",
        9557: "",
        8840: "/18",
        9028: "/18",
    }
    min_max = "min"
    threshold = "3"

    return get_lab_labels(c, concept_ids, unit_concept_ids, min_max, threshold)


def labeler_anemia(c):
    concept_ids: list = [37072252]
    unit_concept_ids = {
        8840: "/100",
        8636: "",
        8713: "*10",
    }
    min_max = "min"
    threshold = "70"

    return get_lab_labels(c, concept_ids, unit_concept_ids, min_max, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeler")
    parser.add_argument("path_to_output_dir", type=str)

    parser.add_argument(
        "--labeler",
        type=str,
        help="Name of labeling function to create.",
        choices=LABELERS,
        default=LABELERS[0],
    )

    parser.add_argument("--cohort_path", type=str, default="data/cohort/ip_cohort")
    parser.add_argument("--refresh_cohort", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    PATH_TO_OMOP: str = path_omop
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_COHORT: str = os.path.join(path_root, args.cohort_path)

    START_TIME = time.time()

    # configure labelers
    if args.labeler == "mortality":
        labeler = labeler_mortality
    elif args.labeler == "long_los":
        labeler = labeler_los
    elif args.labeler == "readmission":
        labeler = labeler_readmission
    elif args.labeler == "thrombocytopenia_lab":
        labeler = labeler_thrombocytopenia
    elif args.labeler == "hyperkalemia_lab":
        labeler = labeler_hyperkalemia
    elif args.labeler == "hypoglycemia_lab":
        labeler = labeler_hypoglycemia
    elif args.labeler == "hyponatremia_lab":
        labeler = labeler_hyponatremia
    elif args.labeler == "anemia_lab":
        labeler = labeler_anemia
    else:
        raise ValueError(
            f"Labeler `{args.labeler}` not supported. Must be one of: {LABELERS}."
        )

    print(
        f"\n\
    OMOP path: {PATH_TO_OMOP}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Cohort path: {PATH_TO_COHORT}\n\
    Labeler: {args.labeler}\n\
    "
    )

    if args.overwrite and os.path.exists(PATH_TO_OUTPUT_DIR):
        shutil.rmtree(PATH_TO_OUTPUT_DIR, ignore_errors=True)

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # register OMOP tables
    c = Context()
    c = register_omop_tables(c, PATH_TO_OMOP)

    # register custom date functions
    c.register_function(
        date_diff_years,
        "date_diff_years",
        [("a", np.datetime64), ("b", np.datetime64)],
        np.float32,
        replace=True,
    )

    c.register_function(
        date_diff_days,
        "date_diff_days",
        [("a", np.datetime64), ("b", np.datetime64)],
        int,
        replace=True,
    )

    c.register_function(
        convert_to_date,
        "convert_to_date",
        [("a", np.datetime64)],
        np.datetime64,
        replace=True,
    )

    # create inpatient cohort
    if args.refresh_cohort and os.path.exists(PATH_TO_COHORT):
        shutil.rmtree(PATH_TO_COHORT, ignore_errors=True)

    if os.path.exists(PATH_TO_COHORT):
        print("Loading cohort")

        df_cohort = pd.read_csv(
            os.path.join(PATH_TO_COHORT, "cohort.csv"),
            parse_dates=["admit_datetime", "discharge_datetime", "index_datetime"],
            dtype={"person_id": "Int64"},
        )

    else:
        os.makedirs(PATH_TO_COHORT, exist_ok=True)
        print("Creating cohort")
        df_cohort = create_ip_cohort(c)
        df_cohort.to_csv(os.path.join(PATH_TO_COHORT, "cohort.csv"), index=False)

    c.create_table("cohort", df_cohort)
    print("Created inpatient cohort")

    # label
    df_labels = labeler(c)
    df_labels.to_csv(
        os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv"), index=False
    )
    print("Created labels")

    time_elapsed = int(time.time() - START_TIME)
    print(f"Finished in {time_elapsed} seconds")
