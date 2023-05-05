import argparse
import datetime
import os
import pickle
import shutil
import time
import subprocess

import numpy as np

from typing import Dict, Callable
from sklearn.metrics import roc_auc_score, average_precision_score

from src.io import save_to_pkl, read_pkl, read_features
from src.utils import hash_pids

"""
TODO: add expected calibration error
"""

metrics = {
    "auroc": roc_auc_score, 
    "auprc": average_precision_score,
}


def run_bootstrap(
    labels: np.array,
    preds: np.array,
    metrics: Dict[str, Callable[[np.array, np.array], float]],
    n_boots: int,
):
    results = {metric+"_bootstrap":[] for metric,_ in metrics.items()}
    n_samples = len(labels)
    np.random.seed(97)
    
    for i in range(n_boots):
        ids = np.random.choice(n_samples, n_samples)
        
        for metric in metrics:
            results[metric+"_bootstrap"].append(
                metrics[metric](labels[ids], preds[ids])
            )

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate adapter model")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_model", type=str)
    parser.add_argument("--n_boots", type=int, default=None, help="num bootstrap iterations")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    START_TIME = time.time()
    
    print(f"\n\
    output_dir: {args.path_to_output_dir}\n\
    path_to_model: {args.path_to_model}\n\
    n_boots: {args.n_boots}\n\
    ")
    
    if args.overwrite and os.path.exists(args.path_to_output_dir):
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)
        
    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

        # load model and metadata
        print("loading model and metadata")
        m = read_pkl(os.path.join(args.path_to_model, "model.pkl"))
        m_info = read_pkl(os.path.join(args.path_to_model, "model_info.pkl"))

        # load features    
        print("loading features")
        X_test, y_test = read_features(
            m_info['path_to_patient_database'],
            m_info['path_to_features'],
            m_info['path_to_labels'],
            m_info['feature_type'],
            is_eval=True,
        )

        preds = m.predict_proba(X_test)[:,1]

        # evaluate
        results = {
            "labels": y_test,
            "predictions": preds,
            "model": m,
        }

        print("evaluating model")
        for metric in metrics:
            results[metric] = metrics[metric](y_test, preds)

        if args.n_boots is not None:
            print(f"evaluating using {args.n_boot} bootstrap iterations")
            boot_results = run_bootstrap(
                y_test,
                preds,
                metrics,
                args.n_boots,
            )

            results = {**results, **boot_results}

        # save
        print("saving results")
        save_to_pkl(results, os.path.join(args.path_to_output_dir, "results.pkl"))

        t_end = int(time.time()-START_TIME)
        print(f"finished in {t_end} seconds")