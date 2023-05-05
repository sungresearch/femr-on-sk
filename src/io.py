import os
import msgpack
import yaml
import pickle
import pdb
import warnings

import numpy as np
from typing import Optional
from femr.datasets import PatientDatabase
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
    with open(path_yaml, 'r') as stream:
        return yaml.safe_load(stream)
    

def read_features(
    path_to_patient_database: str,
    path_to_features: str, 
    path_to_labels: str,
    feature_type: str,
    is_train: bool = False,
    is_eval: bool = False,
    train_perc: Optional[int] = 85,
    train_n: Optional[int] = None,
):  
    """
    Load features and labels for training and evaluation of adapter models.
    
    When train_n is specified (e.g., for few_shot experiments), balanced sampling 
    is additionally conducted to obtain the labels (and corresponding features). 
    """
    if is_train and is_eval:
        raise ValueError("Specify True for only one of is_train or is_eval")
        
    if not is_train and not is_eval:
        raise ValueError("Must specify true for one of is_train or is_eval")
        
    if is_train and train_perc is None:
        raise ValueError("Must specify either train_perc (default uses 85)")
        
    if train_perc is not None and train_perc > 85:
        raise ValueError("You specified over 85% of patients for training, which can cause data leakage")
        
    TEST_SPLIT = 85
    
    labels_object = read_pkl(path_to_labels)
    features = read_pkl(path_to_features)
    
    if feature_type == "clmbr":
        patient_ids = features['patient_ids']
        feature_matrix = features['data_matrix']
    elif feature_type == "count":
        patient_ids = features[1]
        feature_matrix = features[0]
    
    # get hashed patient IDs to generate splits
    hashed_pids = hash_pids(path_to_patient_database, patient_ids)
    
    if is_train: 
        sel_pids_idx = np.where(hashed_pids < train_perc)[0]
        
    if is_eval:
        sel_pids_idx = np.where(hashed_pids >= TEST_SPLIT)[0]
        
    sel_pids = patient_ids[sel_pids_idx]
    X = feature_matrix[sel_pids_idx, :]
    y = np.array([
        labels_object.get_labels_from_patient_idx(x)[0].value
        for x in sel_pids
    ])
    
    if is_train and train_n is not None:
        y_pos = np.where(y==1)[0]
        y_neg = np.where(y==0)[0]
        
        if len(y_pos) < train_n/2 or len(y_neg) < train_n/2:
            warnings.warn(f"Number of samples in one class is < {train_n/2}")
        
        y_index = np.concatenate(
            (
                np.random.choice(y_pos,min(int(train_n/2), len(y_pos))),
                np.random.choice(y_neg,min(int(train_n/2), len(y_neg))),
            )
        )
        np.random.shuffle(y_index)
        
        y = y[y_index]
        X = X[y_index]
    
    return (X, y)