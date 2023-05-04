"""
In this sanity check, we drop top n% of CLMBR vocabulary from 20 to 99%.
The expectation is that CLMBR performance should drop as n increases.
"""

import argparse
import datetime
import os
import pickle
import shutil
import time
import subprocess
import pdb

from typing import List
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from src.utils import (
    save_to_pkl, read_pkl, read_msgpack, save_to_msgpack, 
    load_features, hash_pids
)
    
    
def run_clmbr_featurizer(args, drop_perc: int):
    """
    Featurize using CLMBR
    """
      
    PATH_TO_PATIENT_DATABASE: str = args.path_to_patient_database
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_LABELS: str = args.path_to_labels
    
    PATH_TASK_BATCHES = os.path.join(PATH_TO_OUTPUT_DIR, f"task_batches_{drop_perc}")
    PATH_FEATURES = os.path.join(PATH_TO_OUTPUT_DIR, f"featurized_patients_{drop_perc}.pkl")
    PATH_DICTIONARY = os.path.join(args.path_to_clmbr_data, "dictionary")
    PATH_ABLATED_DICTIONARY = os.path.join(args.path_to_clmbr_data, "ablated_dictionary") 
    PATH_MODEL = os.path.join(args.path_to_clmbr_data, "clmbr_model")
    
    print(f"\n\
    PatientDatabase path: {PATH_TO_PATIENT_DATABASE}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Labels path: {PATH_TO_LABELS}\n\
    CLMBR data path: {args.path_to_clmbr_data}\n\
    ")
    
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    
    # load model config to get vocab size
    model_config = read_msgpack(os.path.join(PATH_MODEL, "config.msgpack"))
    vocab_size = model_config["transformer"]["vocab_size"]
    
    # Load dictionary, modify, write to new dictionary
    dictionary = read_msgpack(PATH_DICTIONARY)
    dictionary["regular"] = dictionary["regular"][:vocab_size]
    
    if drop_perc<100:
        for i in range(int(drop_perc/100*vocab_size)):
            # type 3 disables dictionary entry
            dictionary["regular"][i+1]['type'] = 3  
    
    elif drop_perc==100:
        for i in range(vocab_size-1):
            # drop all except first code, which is birthdate - needed
            dictionary["regular"][i+1]['type'] = 3  
    
    else: 
        raise ValueError("drop_perc cannot be > 100")
            
    save_to_msgpack(dictionary, PATH_ABLATED_DICTIONARY)
    
    # create task batches
    if args.overwrite and os.path.exists(PATH_TASK_BATCHES):
        shutil.rmtree(PATH_TASK_BATCHES, ignore_errors=True)
        
    if not os.path.exists(PATH_TASK_BATCHES):
        cmd = [
            "clmbr_create_batches",
            PATH_TASK_BATCHES, 
            "--data_path", PATH_TO_PATIENT_DATABASE,
            "--dictionary", PATH_ABLATED_DICTIONARY,
            "--task", "labeled_patients",
            "--labeled_patients_path", os.path.join(PATH_TO_LABELS, "labeled_patients.pkl"),
            "--transformer_vocab_size", str(vocab_size)
        ]

        subprocess.run(cmd)

    # compute representations
    if args.overwrite and os.path.exists(PATH_FEATURES):
        os.remove(PATH_FEATURES)
    
    if not os.path.exists(PATH_FEATURES):
        subprocess.run(
            [
                "clmbr_compute_representations",
                PATH_FEATURES, 
                "--data_path", PATH_TO_PATIENT_DATABASE,
                "--batches_path", PATH_TASK_BATCHES,
                "--model_dir", PATH_MODEL,
            ]
        )


def train_eval_adapter(args, drop_perc: int):

    PATH_TO_LABELS: str = os.path.join(args.path_to_labels,"labeled_patients.pkl")
    PATH_FEATURES = os.path.join(args.path_to_output_dir, f"featurized_patients_{drop_perc}.pkl")
    
    X_train, y_train = load_features(
        args.path_to_patient_database,
        PATH_FEATURES,
        PATH_TO_LABELS,
        "clmbr",
        is_train=True,
    )
    
    X_test, y_test = load_features(
        args.path_to_patient_database,
        PATH_FEATURES,
        PATH_TO_LABELS,
        "clmbr",
        is_eval=True,
    )
    
    m = LogisticRegressionCV(
        Cs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 
        cv = 5,
        scoring = "neg_log_loss",
        max_iter = 10000,
        n_jobs = 4,
        refit = True
    )
    
    print("fitting model")
    m.fit(X_train, y_train)
    
    print("saving model")
    save_to_pkl(m, os.path.join(args.path_to_output_dir, f"model_{drop_perc}.pkl"))
    
    print("evaluating model")
    preds = m.predict_proba(X_test)[:,1]
    results = {
        "labels": y_test,
        "predictions": preds,
        "model": m,
        "auroc": roc_auc_score(y_test, preds)
    }
    
    print("saving results")
    save_to_pkl(results, os.path.join(args.path_to_output_dir, f"results_{drop_perc}.pkl"))
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run featurizer")
    parser.add_argument(
        "--path_to_patient_database",
        type=str,
        help="Path to femr PatientDatabase.",
        default="/hpf/projects/lsung/data/lguo/omop_extract_v6",
    )

    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        help=("Path to save features"),
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/stanford_vocab_ablation"
    )

    parser.add_argument(
        "--path_to_labels",
        type=str,
        help="Path to labels.",
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los"
    )
    
    parser.add_argument(
        "--path_to_sk_dictionary",
        type=str,
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_sk/CLMBR_learning_rate_1e-4_max_iter_1000000_rotary_type_per_head/dictionary"
    )

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--path_to_clmbr_data", type=str, default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_stanford")
    args = parser.parse_args()
    
    for drop_perc in [20,40,60,80,95,99,100]:
        run_clmbr_featurizer(args, drop_perc=drop_perc)
        train_eval_adapter(args, drop_perc=drop_perc)