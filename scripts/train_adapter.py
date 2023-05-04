import argparse
import datetime
import os
import pickle
import shutil
import time
import subprocess
import pdb

import numpy as np

from typing import List
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from src.utils import save_to_pkl, read_pkl, load_features, hash_pids
from src.default_paths import path_extract

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train adapter model")
    parser.add_argument(
        "path_to_output_dir",
        type=str,
        help=("Path to save models"
        ),
    )

    parser.add_argument(
        "--path_to_labels",
        type=str,
        help="Path to labels.",
    )
    
    parser.add_argument(
        "--path_to_features",
        type=str,
        help="Path to labels.",
    )
    
    parser.add_argument(
        "--feature_type",
        type=str,
        help="count or clmbr",
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default = 4,
        help="n_jobs for sklearn LogisticRegressionCV"
    )
    
    parser.add_argument(
        "--train_perc",
        type=int,
        default=85, # default split is 85:15
        help="perc training patients [int]"
    )
    
    parser.add_argument(
        "--train_n",
        type=int,
        default=None, # default split is 85:15
        help="N training patients [int]"
    )

    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    PATH_TO_FEATURES = os.path.join(args.path_to_features, "featurized_patients.pkl")
    PATH_TO_LABELS = os.path.join(args.path_to_labels, "labeled_patients.pkl")
    START_TIME = time.time()
    
    print(f"\n\
    output_dir: {args.path_to_output_dir}\n\
    path_to_features: {PATH_TO_FEATURES}\n\
    path_to_labels: {PATH_TO_LABELS}\n\
    path_to_patient_database: {path_extract}\n\
    feature_type: {args.feature_type}\n\
    percent patients for training: {args.train_perc}%\n\
    N patients for training: {args.train_n}\n\
    ")
    
    
    if args.overwrite and os.path.exists(args.path_to_output_dir):
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)
        
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # load features
    if args.feature_type not in ["count", "clmbr"]:
        raise ValueError("--feature_type must be 'count' or 'clmbr'")
    
    X_train, y_train = load_features(
        path_extract,
        PATH_TO_FEATURES,
        PATH_TO_LABELS,
        args.feature_type,
        is_train=True,
        train_perc=args.train_perc,
        train_n=args.train_n,
    )
        
    # fit logistic regression with 5-fold cross validation
    m = LogisticRegressionCV(
        Cs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 
        cv = 5,
        scoring = "neg_log_loss",
        max_iter = 10000,
        n_jobs = args.n_jobs,
        refit = True
    )
    
    if args.train_n is not None:
        # if in few_shots setting, use 1e-1 by default
        m = LogisticRegression(C=0.1, max_iter=10000, n_jobs=args.n_jobs)
    
    print("fitting model")
    m.fit(X_train, y_train)
    
    print("saving models")
    # save model
    save_to_pkl(m, os.path.join(args.path_to_output_dir, "model.pkl"))
    
    m_info = {
        "path_to_patient_database": path_extract,
        "path_to_features": PATH_TO_FEATURES, 
        "path_to_labels": PATH_TO_LABELS,
        "feature_type": args.feature_type,
        "train_perc": args.train_perc,
    }
    
    save_to_pkl(m_info, os.path.join(args.path_to_output_dir, "model_info.pkl"))
    
    t_end = int(time.time()-START_TIME)
    print(f"finished in {t_end} seconds")