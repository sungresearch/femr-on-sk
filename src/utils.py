import os
import numpy as np
from femr.datasets import PatientDatabase


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
