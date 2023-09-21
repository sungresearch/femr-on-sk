import os
import random
import numpy as np
import pandas as pd

from femr.datasets import PatientDatabase
from src.default_paths import path_extract, path_root


def list_dir(path: str):
    """get list of file/directory names excluding nb checkpoints"""

    return [x for x in os.listdir(path) if x != ".ipynb_checkpoints"]


def hash_pids(path_to_patient_database: str, patient_ids: np.array):
    """
    We are hashing PIDs using the same seed as the create_batches process
    to deterministically create 2-digit ints that will be used to assign
    patients into train/test sets.
    """
    database = PatientDatabase(path_to_patient_database)
    return np.array([database.compute_split(97, pid) for pid in patient_ids])


def get_best_clmbr_model(path):
    best_model = ""
    best_score = 99999

    for model in list_dir(path):
        if not os.path.isdir(os.path.join(path, model)):
            continue

        with open(os.path.join(path, model, "clmbr_model", "best_info")) as f:
            message = f.read()
            score = float(
                [x.split(":") for x in message.split(",") if "'loss'" in x][0][1]
            )

            if score < best_score:
                best_score = score
                best_model = model

    return best_model


def create_restricted_patients_file(
    output_dir: str,
    percentage: float,
    seed: int = 444,
    overwrite: bool = True,
):
    """deterministically create restricted patient IDs file"""
    if overwrite and os.path.exists(output_dir):
        os.remove(output_dir)

    if not os.path.exists(output_dir):
        pids = list(PatientDatabase(path_extract))
        random.Random(seed).shuffle(pids)

        n_patients = int(len(pids) * percentage)
        sel_pids = pids[:n_patients]

        with open(output_dir, "w") as f:
            for pid in sel_pids:
                f.write(f"{pid}\n")


def create_restricted_patients_file_few_shots(
    output_dir: str,
    cohort_dir: str = None,
    exclude_percentage: float = 0.2,
    seed: int = 444,
    overwrite: bool = True,
):
    """
    deterministically create restricted patient IDs file that include all patient IDs
    except the ones sampled
    """
    if not cohort_dir:
        cohort_dir = os.path.join(path_root, "data/cohort/ip_cohort/cohort.csv")

    if overwrite and os.path.exists(output_dir):
        os.remove(output_dir)

    if not os.path.exists(output_dir):
        df_cohort = pd.read_csv(cohort_dir)
        pids = df_cohort.person_id.values
        hashed_pids = hash_pids(path_extract, pids)
        pid_to_hash_dict = {k: hashed_pids[i] for i, k in enumerate(pids)}

        pid_to_hash_df = pd.DataFrame.from_dict(
            pid_to_hash_dict, orient="index", columns=["hash"]
        ).reset_index(names=["person_id"])

        df_cohort = df_cohort.merge(
            pid_to_hash_df.query("hash < 85"), how="inner", on="person_id"
        )

        pids_to_exclude = df_cohort.sample(
            frac=exclude_percentage, replace=False, random_state=seed
        )["person_id"].values

        all_pids = list(PatientDatabase(path_extract))
        sel_pids = [x for x in all_pids if x not in pids_to_exclude]

        print(len(pids_to_exclude), len(all_pids), len(sel_pids))

        with open(output_dir, "w") as f:
            for pid in sel_pids:
                f.write(f"{pid}\n")
