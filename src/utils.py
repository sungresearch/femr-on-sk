import os
import random
import numpy as np
from femr.datasets import PatientDatabase
from src.default_paths import path_extract


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
