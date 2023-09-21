import argparse
import os
import shutil
import time
import functools
import multiprocessing
import sys

import numpy as np
import pandas as pd

from typing import List, Dict, Callable, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample

from src.io import save_to_pkl, read_pkl
from src.metrics import average_precision_score_calibrated, expected_calibration_error

tasks = [
    "long_los_sql",
    "mortality_sql",
    "readmission_sql",
    "hyponatremia_lab_sql",
    "hyperkalemia_lab_sql",
    "anemia_lab_sql",
    "hypoglycemia_lab_sql",
    "thrombocytopenia_lab_sql",
]


def _run_boot(args: Tuple[int, Dict, List, Dict, Dict]) -> pd.DataFrame:
    """
    Run a single hierarchical bootstrap iteration, return results in dataframe
    This function is supposed to run with a multiprocess pool.
    """
    (iboot, metrics, tasks, m1_results, m2_results) = args

    out = pd.DataFrame()

    # sample tasks
    selected_tasks = np.random.choice(tasks, size=len(tasks), replace=True)
    for task in selected_tasks:
        m1_task_results = m1_results[task]
        m2_task_results = m2_results[task]

        # sample patients
        # the usage of resample below requires that all arrays are equal length and identically ordered by patient ID
        m1_preds, m1_labels, m2_preds, m2_labels = resample(
            m1_task_results["predictions"],
            m1_task_results["labels"],
            m2_task_results["predictions"],
            m2_task_results["labels"],
            replace=True,
            n_samples=len(m1_task_results["patient_ids"]),
        )

        # get performance
        for metric in metrics:
            m1_metric = metrics[metric](m1_labels, m1_preds)
            m2_metric = metrics[metric](m2_labels, m2_preds)

            out = pd.concat(
                (
                    out,
                    pd.DataFrame(
                        {
                            "boot_iter": [iboot],
                            "task": task,
                            "metric": metric,
                            "m1": m1_metric,
                            "m2": m2_metric,
                            "difference": m1_metric - m2_metric,
                        }
                    ),
                )
            )

    return out


def _run_boot_with_iters(args: Tuple[int, Dict, int, List, Dict, Dict]) -> pd.DataFrame:
    """
    Run a single hierarchical bootstrap iteration, return results in dataframe
    This function is supposed to run with a multiprocess pool.

    Difference between this and _run_boot() is that this expects multiple models
    were trained for a given task and model (i.e., in the few shots setting)
    """
    (iboot, metrics, n_model_train_iters, tasks, m1_results, m2_results) = args

    out = pd.DataFrame()

    # sample tasks
    selected_tasks = np.random.choice(tasks, size=len(tasks), replace=True)
    for task in selected_tasks:
        m1_task_results = m1_results[task]
        m2_task_results = m2_results[task]

        for metric in metrics:
            c = 0
            m1_metric, m2_metric = 0, 0

            for i_iter in range(n_model_train_iters):
                try:
                    # sample patients
                    # below requires all arrays to be equal length and identically ordered by patient ID
                    m1_preds, m1_labels, m2_preds, m2_labels = resample(
                        m1_task_results[i_iter]["predictions"],
                        m1_task_results[i_iter]["labels"],
                        m2_task_results[i_iter]["predictions"],
                        m2_task_results[i_iter]["labels"],
                        replace=True,
                        n_samples=len(m1_task_results[i_iter]["patient_ids"]),
                    )

                    m1_metric += metrics[metric](m1_labels, m1_preds)
                    m2_metric += metrics[metric](m2_labels, m2_preds)

                    c += 1
                except (KeyError, ZeroDivisionError):
                    continue

            m_m1_metric, m_m2_metric = np.nan, np.nan

            if c > 0:
                m_m1_metric = m1_metric / c
                m_m2_metric = m2_metric / c

            out = pd.concat(
                (
                    out,
                    pd.DataFrame(
                        {
                            "boot_iter": [iboot],
                            "task": task,
                            "metric": metric,
                            "m1": m_m1_metric,
                            "m2": m_m2_metric,
                            "difference": m_m1_metric - m_m2_metric,
                        }
                    ),
                )
            )

    return out


def _get_pvalue(df: pd.DataFrame) -> float:
    """Two-tailed p-value computed by taking 2*min(p, 1−p) while taking into account values
    that were equal to the threshold (0 in this case).
    """
    if "difference" not in df.columns:
        raise KeyError(
            "`difference` must be a column that contains difference in the metric"
        )

    d = df["difference"].values
    return (2 * min((d <= 0).sum(), (d >= 0).sum()) - (d == 0).sum()) / len(d)


def hb_tasks(
    path_to_results: str,
    model_1_name: str,
    model_2_name: str,
    force_model_2_path: str = None,
    metrics: Dict[str, Callable] = None,
    tasks: List[str] = tasks,
    n_core: int = 10,
    chunksize: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Conduct hierarchical bootstrap comparing model 1 with model 2 by bootstrapping patients and tasks.
    """
    m1_results = {}
    m2_results = {}
    for task in tasks:
        m1_results[task] = read_pkl(
            os.path.join(path_to_results, model_1_name, task, "results.pkl")
        )

        if force_model_2_path:
            m2_results[task] = read_pkl(
                os.path.join(force_model_2_path, task, "results.pkl")
            )

        else:
            m2_results[task] = read_pkl(
                os.path.join(path_to_results, model_2_name, task, "results.pkl")
            )

        # asserts that arrays are identically ordered by patient ID and equal length
        assert (
            m1_results[task]["patient_ids"] == m2_results[task]["patient_ids"]
        ).sum() == len(set(m1_results[task]["patient_ids"]))

    # conduct bootstrap
    raw_results = []
    with multiprocessing.Pool(n_core) as p:
        for i, r in enumerate(
            p.imap_unordered(
                _run_boot,
                (
                    (iboot, metrics, tasks, m1_results, m2_results)
                    for iboot in range(args.n_boots)
                ),
                chunksize,
            )
        ):
            if i % 100 == 0:
                sys.stderr.write("\rdone {0:%}".format(i / args.n_boots))

            raw_results.append(r)

    raw_results = pd.concat((raw_results)).rename(
        columns={"m1": model_1_name, "m2": model_2_name}
    )

    # agg results
    agg_results = (
        raw_results.groupby(["boot_iter", "metric"])[
            [model_1_name, model_2_name, "difference"]
        ]
        .mean()
        .reset_index()
    )

    # construct percentile bootstrap CI
    results_ci = agg_results.groupby("metric")[
        [model_1_name, model_2_name, "difference"]
    ].quantile([0.025, 0.5, 0.975])

    # get boostrap p-value
    results_p = (
        agg_results.groupby("metric")
        .apply(_get_pvalue)
        .reset_index()
        .rename(columns={0: "p-value"})
    )

    # get actual means for each model for reference
    m1_results_mean, m2_results_mean = {}, {}
    for metric in metrics.keys():
        m1_results_mean[metric], m2_results_mean[metric] = 0, 0

        for task in tasks:
            if metric not in m1_results[task].keys():
                raise KeyError(
                    f"{metric} not found for task {task} for model {model_1_name}"
                )

            if metric not in m2_results[task].keys():
                raise KeyError(
                    f"{metric} not found for task {task} for model {model_1_name}"
                )

            m1_results_mean[metric] += m1_results[task][metric]
            m2_results_mean[metric] += m2_results[task][metric]

        m1_results_mean[metric] /= len(tasks)
        m2_results_mean[metric] /= len(tasks)

    return {
        "model_1": model_1_name,
        "model_2": model_2_name,
        "results_bootstrap": raw_results,
        "results_ci": results_ci,
        "results_p": results_p,
        f"{model_1_name}_mean": m1_results_mean,
        f"{model_2_name}_mean": m2_results_mean,
    }


def hb_tasks_with_iters(
    path_to_results: str,
    model_1_name: str,
    model_2_name: str,
    n_model_train_iters: int,
    force_model_2_path: str = None,
    metrics: Dict[str, Callable] = None,
    tasks: List[str] = tasks,
    n_core: int = 10,
    chunksize: int = 100,
) -> Dict[str, pd.DataFrame]:
    m1_results = {}
    m2_results = {}
    for task in tasks:
        m1_results[task] = {}
        m2_results[task] = {}

        for i_iter in range(n_model_train_iters):
            try:
                m1_results[task][i_iter] = read_pkl(
                    os.path.join(
                        path_to_results,
                        f"{model_1_name}_iter{i_iter}",
                        task,
                        "results.pkl",
                    )
                )
            except FileNotFoundError:
                print(f"{i_iter} skipped for {task} in {model_1_name}")
                pass

            try:
                if force_model_2_path:
                    m2_results[task][i_iter] = read_pkl(
                        os.path.join(force_model_2_path, task, "results.pkl")
                    )
                else:
                    m2_results[task][i_iter] = read_pkl(
                        os.path.join(
                            path_to_results,
                            f"{model_2_name}_iter{i_iter}",
                            task,
                            "results.pkl",
                        )
                    )
            except FileNotFoundError:
                print(f"{i_iter} skipped for {task} in {model_2_name}")
                pass

            if i_iter in m1_results[task] and i_iter in m2_results[task]:
                # asserts that arrays are identically ordered by patient ID and equal length
                assert (
                    m1_results[task][i_iter]["patient_ids"]
                    == m2_results[task][i_iter]["patient_ids"]
                ).sum() == len(set(m1_results[task][i_iter]["patient_ids"]))

    # conduct bootstrap
    raw_results = []
    with multiprocessing.Pool(n_core) as p:
        for i, r in enumerate(
            p.imap_unordered(
                _run_boot_with_iters,
                (
                    (iboot, metrics, n_model_train_iters, tasks, m1_results, m2_results)
                    for iboot in range(args.n_boots)
                ),
                chunksize,
            )
        ):
            if i % 100 == 0:
                sys.stderr.write("\rdone {0:%}".format(i / args.n_boots))

            raw_results.append(r)

    raw_results = pd.concat((raw_results)).rename(
        columns={"m1": model_1_name, "m2": model_2_name}
    )

    # agg results
    agg_results = (
        raw_results.groupby(["boot_iter", "metric"])[
            [model_1_name, model_2_name, "difference"]
        ]
        .mean()
        .reset_index()
    )

    # construct percentile bootstrap CI
    results_ci = agg_results.groupby("metric")[
        [model_1_name, model_2_name, "difference"]
    ].quantile([0.025, 0.5, 0.975])

    # get boostrap p-value
    results_p = (
        agg_results.groupby("metric")
        .apply(_get_pvalue)
        .reset_index()
        .rename(columns={0: "p-value"})
    )

    # get actual means for each model for reference
    m1_results_mean, m2_results_mean = {}, {}
    for metric in metrics.keys():
        m1_results_mean[metric], m2_results_mean[metric] = 0, 0

        for task in tasks:
            c = 0
            m1_metric, m2_metric = 0, 0
            for i_iter in range(n_model_train_iters):
                if i_iter not in m1_results[task] or i_iter not in m2_results[task]:
                    continue

                if metric not in m1_results[task][i_iter].keys():
                    raise KeyError(
                        f"{metric} not found for task {task} for model {model_1_name}"
                    )

                if metric not in m2_results[task][i_iter].keys():
                    raise KeyError(
                        f"{metric} not found for task {task} for model {model_1_name}"
                    )

                m1_metric += m1_results[task][i_iter][metric]
                m2_metric += m2_results[task][i_iter][metric]
                c += 1

            if c == 0:
                raise ValueError(
                    f"{task} for {model_1_name} or {model_2_name} does not contain results"
                )

            m1_results_mean[metric] += m1_metric / c
            m2_results_mean[metric] += m2_metric / c

        m1_results_mean[metric] /= len(tasks)
        m2_results_mean[metric] /= len(tasks)

    return {
        "model_1": model_1_name,
        "model_2": model_2_name,
        "results_bootstrap": raw_results,
        "results_ci": results_ci,
        "results_p": results_p,
        f"{model_1_name}_mean": m1_results_mean,
        f"{model_2_name}_mean": m2_results_mean,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conduct hierarchical bootstrap comparing two models"
    )
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument(
        "--path_to_results", type=str, help="path to evaluation results"
    )
    parser.add_argument("--model_1", type=str, help="model 1 name")
    parser.add_argument("--model_2", type=str, help="model 2 name")
    parser.add_argument("--n_boots", type=int, default=10000)
    parser.add_argument("--correct_calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--n_model_train_iters",
        type=int,
        default=None,
        help="\
            include the number of iterations for which models were trained (i.e., in the few_shots experiment) \
            as another level for hierarichal bootstrapping",
    )
    parser.add_argument(
        "--force_model_2_path",
        type=str,
        default=None,
        help="\
            hacky patch to force use clmbr_stanford results from a different results folder for the subsample \
            experiment.",
    )
    args = parser.parse_args()

    metrics = {
        "auroc": roc_auc_score,
        "auprc": average_precision_score,
        "auprc_c": average_precision_score_calibrated,
        "ece": functools.partial(
            expected_calibration_error,
            num_bins=10,
            quantile_bins=True,
            correct_y_pred=args.correct_calibration,
            b0=0.5,  # only used when args.correct_calibration = True
        ),
    }

    START_TIME = time.time()
    np.random.seed(args.seed)

    if os.path.exists(args.path_to_output_dir) and args.overwrite:
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

        if args.n_model_train_iters is None:
            out = hb_tasks(
                path_to_results=args.path_to_results,
                model_1_name=args.model_1,
                model_2_name=args.model_2,
                force_model_2_path=args.force_model_2_path,
                metrics=metrics,
            )
        else:
            out = hb_tasks_with_iters(
                path_to_results=args.path_to_results,
                model_1_name=args.model_1,
                model_2_name=args.model_2,
                n_model_train_iters=args.n_model_train_iters,
                force_model_2_path=args.force_model_2_path,
                metrics=metrics,
            )

        # save
        print("saving results")
        save_to_pkl(out, os.path.join(args.path_to_output_dir, "results.pkl"))

        t_end = int(time.time() - START_TIME)
        print(f"{args.model_1} vs {args.model_2}")
        print(out["results_ci"], out["results_p"])

    print(f"finished in {t_end} seconds")
