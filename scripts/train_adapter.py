import argparse
import os
import shutil
import time
import scipy

import numpy as np

from lightgbm import LGBMClassifier as gbm

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument(
        "--patients_file_to_exclude",
        type=str,
        default=None,
        help="Path to file containing patient IDs to be excluded for few_shot experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        help="logistic_regression/lightgbm",
    )
    parser.add_argument("--scale_features", action="store_true")
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
    feature_type: {args.feature_type}\n\
    N patients for training: {args.train_n}\n\
    "
    )

    if args.model not in ["logistic_regression", "lightgbm"]:
        raise ValueError("model must be either logistic_regression or lightgbm")

    if args.overwrite and os.path.exists(args.path_to_output_dir):
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

        # load features
        if args.feature_type not in ["count", "clmbr", "motor"]:
            raise ValueError("--feature_type must be 'count' or 'clmbr' or 'motor'")

        X_train, y_train, X_val, y_val = read_features(
            path_extract,
            PATH_TO_FEATURES,
            PATH_TO_LABELS,
            args.feature_type,
            is_train=True,
            train_n=args.train_n,
            patients_file_to_exclude=args.patients_file_to_exclude,
        )

        best_model = None
        best_score = 999999

        if args.model == "logistic_regression":
            print("fitting logistic regression")
            best_l2 = 0
            start_l, end_l = -5, 1

            if type(X_train) == scipy.sparse.csr_matrix and args.scale_features:
                X_train = X_train.toarray()
                X_val = X_val.toarray()

            for l_exp in np.linspace(end_l, start_l, num=20):
                l2 = 10 ** (l_exp)
                print(f"{l2=};")

                if args.scale_features:
                    m = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=l2,
                                    max_iter=10000,
                                    n_jobs=args.n_jobs,
                                ),
                            ),
                        ]
                    )

                else:
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
                    print(f"updated {best_score=}, {best_l2=}")

            m_info = {
                "path_to_patient_database": path_extract,
                "path_to_features": PATH_TO_FEATURES,
                "path_to_labels": PATH_TO_LABELS,
                "feature_type": args.feature_type,
                "best_score": best_score,
                "best_l2": best_l2,
            }

        if args.model == "lightgbm":
            print("fitting lightgbm")
            best_lr = 0
            best_num_leaves = 0
            best_boosting_type = ""

            for lr in [0.01, 0.1, 0.2]:
                for num_leaves in [100, 300]:
                    for boosting_type in ["gbdt", "dart", "goss"]:
                        print(f"{lr=}; {num_leaves=}; {boosting_type=}")

                        m = gbm(
                            learning_rate=lr,
                            num_leaves=num_leaves,
                            n_estimators=1000,
                            max_depth=-1,
                            boosting_type=boosting_type,
                            objective="binary",
                            metric="binary_logloss",
                            first_metric_only=True,
                            min_child_samples=min(20, int(X_train.shape[0] / 2)),
                            n_jobs=args.n_jobs,
                        )

                        m.fit(X_train, y_train)
                        score = log_loss(y_val, m.predict_proba(X_val)[:, 1])

                        if score < best_score:
                            best_score = score
                            best_model = m
                            best_lr = lr
                            best_num_leaves = num_leaves
                            best_boosting_type = boosting_type
                            print(
                                f"updated {best_score=}, {best_lr=}, {best_num_leaves=}, {best_boosting_type=}"
                            )

            m_info = {
                "path_to_patient_database": path_extract,
                "path_to_features": PATH_TO_FEATURES,
                "path_to_labels": PATH_TO_LABELS,
                "feature_type": args.feature_type,
                "best_score": best_score,
                "best_lr": best_lr,
                "best_num_leaves": best_num_leaves,
                "best_boosting_type": best_boosting_type,
            }

        print("saving model")
        # save model
        save_to_pkl(best_model, os.path.join(args.path_to_output_dir, "model.pkl"))
        save_to_pkl(m_info, os.path.join(args.path_to_output_dir, "model_info.pkl"))
        t_end = int(time.time() - START_TIME)
        print(f"finished in {t_end} seconds")
