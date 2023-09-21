import os
import msgpack
import yaml
import pickle
import warnings

import numpy as np
from femr.labelers import load_labeled_patients
from typing import Optional
from src.utils import hash_pids


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to pkl file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)


def save_to_msgpack(object_to_save, path_to_file: str):
    """save to msgpack file"""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        msgpack.dump(object_to_save, fd)


def read_pkl(path_to_file: str):
    """Read from pickle file"""
    with open(path_to_file, "rb") as f:
        return pickle.load(f)


def read_msgpack(path_to_file: str):
    """Read from msgpack"""
    with open(path_to_file, "rb") as f:
        return msgpack.load(f)


def read_yaml(path_yaml: str):
    """Read from yaml file"""
    with open(path_yaml, "r") as stream:
        return yaml.safe_load(stream)


def read_features(
    path_to_patient_database: str,
    path_to_features: str,
    path_to_labels: str,
    feature_type: str,
    is_train: bool = False,
    is_eval: bool = False,
    train_n: Optional[int] = None,
    return_patient_ids: bool = False,
    patients_file_to_exclude: str = None,
):
    """
    Load features and labels for training and evaluation of adapter models.

    Train: [min, 70)
    Val: [70, 85)
    Test: [85, max]

    When train_n is specified (e.g., for few_shot experiments), balanced sampling
    is additionally conducted to obtain the labels (and corresponding features).
    """
    if is_train and is_eval:
        raise ValueError("Specify True for only one of is_train or is_eval")

    if not is_train and not is_eval:
        raise ValueError("Must specify true for one of is_train or is_eval")

    VAL_START = 70
    TEST_START = 85

    labels_object = load_labeled_patients(path_to_labels)
    features = read_pkl(path_to_features)

    if feature_type in ["clmbr", "motor"]:
        patient_ids = features["patient_ids"]
        feature_matrix = features["data_matrix"]
    elif feature_type == "count":
        patient_ids = features[1]
        feature_matrix = features[0]

    # Exclude patients specified in the patients file
    # For few-shots experiment, this is only needed to be done for is_train=True
    # For evaluation (is_eval=True), those patients would have already been excluded
    if patients_file_to_exclude:
        with open(patients_file_to_exclude, "r") as f:
            ids_to_exclude = [int(a) for a in f]

        indices = np.where(~np.isin(patient_ids, ids_to_exclude))[0]
        patient_ids = patient_ids[indices]
        feature_matrix = feature_matrix[indices, :]

    # get hashed patient IDs to generate splits
    hashed_pids = hash_pids(path_to_patient_database, patient_ids)

    if is_eval:
        sel_pids_idx = np.where(hashed_pids >= TEST_START)[0]
        sel_pids = patient_ids[sel_pids_idx]
        X = feature_matrix[sel_pids_idx, :]
        y = np.array(
            [labels_object.get_labels_from_patient_idx(x)[0].value for x in sel_pids]
        )

        if return_patient_ids:
            return (X, y, sel_pids)
        else:
            return (X, y)

    if is_train:
        train_idx = np.where(hashed_pids < VAL_START)[0]
        val_idx = np.where((hashed_pids >= VAL_START) & (hashed_pids < TEST_START))[0]

        train_pids = patient_ids[train_idx]
        X_train = feature_matrix[train_idx, :]
        y_train = np.array(
            [labels_object.get_labels_from_patient_idx(x)[0].value for x in train_pids]
        )

        val_pids = patient_ids[val_idx]
        X_val = feature_matrix[val_idx, :]
        y_val = np.array(
            [labels_object.get_labels_from_patient_idx(x)[0].value for x in val_pids]
        )

        if train_n is not None:
            y_pos = np.where(y_train == 1)[0]
            y_neg = np.where(y_train == 0)[0]

            if len(y_pos) < train_n / 2 or len(y_neg) < train_n / 2:
                warnings.warn(f"Number of samples in one class is < {train_n/2}")

            y_index = np.concatenate(
                (
                    np.random.choice(y_pos, min(int(train_n / 2), len(y_pos))),
                    np.random.choice(y_neg, min(int(train_n / 2), len(y_neg))),
                )
            )
            np.random.shuffle(y_index)

            y_train = y_train[y_index]
            X_train = X_train[y_index, :]
            train_pids = train_pids[y_index]

        if return_patient_ids:
            return (X_train, y_train, train_pids, X_val, y_val, val_pids)
        else:
            return (X_train, y_train, X_val, y_val)
