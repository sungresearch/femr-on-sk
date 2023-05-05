import os
import msgpack
import yaml
import pickle
import pdb
import warnings

import numpy as np
from typing import Optional
from femr.datasets import PatientDatabase


def hash_pids(
    path_to_patient_database: str,
    patient_ids: np.array
):
    """
    We are hashing PIDs using the same seed as the create_batches process
    to deterministically create 2-digit ints that will be used to assign
    patients into train/test sets. 
    """
    database = PatientDatabase(path_to_patient_database)
    return np.array([
        database.compute_split(97, pid) 
        for pid in patient_ids
    ])