import argparse
import os
import pickle
import shutil
import time
from typing import List

from femr.datasets import PatientDatabase
from femr.labelers.core import NLabelsPerPatientLabeler
from femr.labelers.omop_inpatient_admissions import (
    InpatientLongAdmissionLabeler,
    InpatientMortalityLabeler,
    InpatientReadmissionLabeler,
)
from femr.labelers.omop_lab_values import (
    AnemiaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
    HyponatremiaLabValueLabeler,
    ThrombocytopeniaLabValueLabeler,
)

from src.default_paths import path_extract


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


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to pkl file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)


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

    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    parser.add_argument(
        "--max_labels_per_patient",
        type=int,
        help="Max number of labels to keep per patient (excess labels are randomly discarded)",
        default=1,
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE: str = path_extract
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads
    MAX_LABELS_PER_PATIENT: int = args.max_labels_per_patient

    START_TIME = time.time()

    # Load PatientDatabase + Ontology
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()

    # configure labelers
    if args.labeler == "mortality":
        labeler = InpatientMortalityLabeler(ontology)
    elif args.labeler == "long_los":
        labeler = InpatientLongAdmissionLabeler(ontology)
    elif args.labeler == "readmission":
        labeler = InpatientReadmissionLabeler(ontology)
    elif args.labeler == "thrombocytopenia_lab":
        labeler = ThrombocytopeniaLabValueLabeler(ontology, "severe")
    elif args.labeler == "hyperkalemia_lab":
        labeler = HyperkalemiaLabValueLabeler(ontology, "severe")
    elif args.labeler == "hypoglycemia_lab":
        labeler = HypoglycemiaLabValueLabeler(ontology, "severe")
    elif args.labeler == "hyponatremia_lab":
        labeler = HyponatremiaLabValueLabeler(ontology, "severe")
    elif args.labeler == "anemia_lab":
        labeler = AnemiaLabValueLabeler(ontology, "severe")
    else:
        raise ValueError(
            f"Labeler `{args.labeler}` not supported. Must be one of: {LABELERS}."
        )

    labeler = NLabelsPerPatientLabeler(
        labeler, seed=0, num_labels=MAX_LABELS_PER_PATIENT
    )

    print(
        f"\n\
    PatientDatabase path: {PATH_TO_PATIENT_DATABASE}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Max number of labels per patient: {MAX_LABELS_PER_PATIENT}\n\
    Labeler: {args.labeler}\n\
    "
    )

    if args.overwrite and os.path.exists(PATH_TO_OUTPUT_DIR):
        shutil.rmtree(PATH_TO_OUTPUT_DIR, ignore_errors=True)

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    labeled_patients = labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        num_threads=NUM_THREADS,
    )

    labeled_patients.save(os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv"))

    time_elapsed = int(time.time() - START_TIME)
    print(f"Finished in {time_elapsed} seconds")
