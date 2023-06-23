import argparse
import os
import shutil
import time

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.io import save_to_pkl, read_features
from src.default_paths import path_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adapter model")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_labels", type=str)
    parser.add_argument("--path_to_features", type=str)
    parser.add_argument("--feature_type", type=str)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument(
        "--train_n",
        type=int,
        default=None,
        help="N training patients [int] for few_shot experiment",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    PATH_TO_FEATURES = os.path.join(args.path_to_features, "featurized_patients.pkl")
    PATH_TO_LABELS = os.path.join(args.path_to_labels, "labeled_patients.csv")
    START_TIME = time.time()

    print(
        f"\n\
    output_dir: {args.path_to_output_dir}\n\
    path_to_features: {PATH_TO_FEATURES}\n\
    path_to_labels: {PATH_TO_LABELS}\n\
    path_to_patient_database: {path_extract}\n\
    feature_type: {args.feature_type}\n\
    N patients for training: {args.train_n}\n\
    "
    )

    if args.overwrite and os.path.exists(args.path_to_output_dir):
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

        # load features
        if args.feature_type not in ["count", "clmbr"]:
            raise ValueError("--feature_type must be 'count' or 'clmbr'")

        X_train, y_train, X_val, y_val = read_features(
            path_extract,
            PATH_TO_FEATURES,
            PATH_TO_LABELS,
            args.feature_type,
            is_train=True,
            train_n=args.train_n,
        )

        print("fitting model")

        best_model = None
        best_score = 999999
        best_l2 = 0

        start_l, end_l = -5, 1
        for l_exp in np.linspace(end_l, start_l, num=20):
            l2 = 10 ** (l_exp)

            m = LogisticRegression(
                C=l2,
                max_iter=10000,
                n_jobs=args.n_jobs,
            )

            m.fit(X_train, y_train)

            score = log_loss(y_val, m.predict_proba(X_val)[:, 1])

            if score < best_score:
                best_score = score
                best_model = m
                best_l2 = l2

        print("saving model")
        # save model
        save_to_pkl(best_model, os.path.join(args.path_to_output_dir, "model.pkl"))

        m_info = {
            "path_to_patient_database": path_extract,
            "path_to_features": PATH_TO_FEATURES,
            "path_to_labels": PATH_TO_LABELS,
            "feature_type": args.feature_type,
            "best_score": best_score,
            "best_l2": best_l2,
        }

        save_to_pkl(m_info, os.path.join(args.path_to_output_dir, "model_info.pkl"))

        t_end = int(time.time() - START_TIME)
        print(f"finished in {t_end} seconds")
