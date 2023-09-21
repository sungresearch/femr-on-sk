import argparse
import os
import shutil
import time
import random
import pickle
import scipy
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import sklearn.pipeline

import femr.datasets
import femr.extension.dataloader
import femr.models.dataloader
import femr.models.transformer

from typing import Dict, Callable
from sklearn.metrics import roc_auc_score, average_precision_score

from src.io import save_to_pkl, read_pkl, read_features
from src.default_paths import path_extract
from src.utils import hash_pids, get_best_clmbr_model
from src.metrics import average_precision_score_calibrated, expected_calibration_error

metrics = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "auprc_c": average_precision_score_calibrated,
    "ece": functools.partial(
        expected_calibration_error, num_bins=10, quantile_bins=True
    ),
}


def get_adapter_model_results(args):
    print("loading model and metadata")
    m = read_pkl(os.path.join(args.path_to_model, "model.pkl"))
    m_info = read_pkl(os.path.join(args.path_to_model, "model_info.pkl"))

    # load features
    print("loading features")
    X_test, y_test, pids = read_features(
        m_info["path_to_patient_database"],
        m_info["path_to_features"],
        m_info["path_to_labels"],
        m_info["feature_type"],
        is_eval=True,
        return_patient_ids=True,
    )

    if (
        type(X_test) == scipy.sparse.csr_matrix
        and type(m) == sklearn.pipeline.Pipeline
        and "scaler" in m.named_steps.keys()
    ):
        X_test = X_test.toarray()

    preds = m.predict_proba(X_test)[:, 1]

    # evaluate
    results = {
        "data_path": m_info["path_to_patient_database"],
        "model": args.path_to_model,
        "labels": y_test,
        "predictions": preds,
        "patient_ids": pids,
    }

    return results


def get_linear_probe_results(args):
    path_to_results = os.path.join(args.path_to_model, "model/predictions.pkl")

    (predictions, label_pids, label_values, prediction_dates) = read_pkl(
        path_to_results
    )

    hashed_pids = hash_pids(path_extract, label_pids)
    test_idx = np.where(hashed_pids >= 85)

    test_pids = label_pids[test_idx]
    test_preds = predictions[test_idx]
    test_labels = label_values[test_idx]

    results = {
        "data_path": path_extract,
        "model": args.path_to_model,
        "labels": test_labels,
        "predictions": test_preds,
        "patient_ids": test_pids,
    }

    return results


def get_clmbr_task_model_results(args):
    best_model = get_best_clmbr_model(args.path_to_model)
    clmbr_data_path = os.path.join(args.path_to_model, best_model)
    model_path = os.path.join(clmbr_data_path, "clmbr_model")
    batch_info_path = os.path.join(
        clmbr_data_path, "task_batches", "batch_info.msgpack"
    )

    print("loading model and batches")
    with open(os.path.join(model_path, "config.msgpack"), "rb") as f:
        config = msgpack.load(f, use_list=False)

    random.seed(config["seed"])

    config = hk.data_structures.to_immutable_dict(config)

    loader = femr.extension.dataloader.BatchLoader(path_extract, batch_info_path)

    def model_fn(config, batch):
        model = femr.models.transformer.EHRTransformer(config)(batch, is_training=False)
        return model

    dummy_batch = loader.get_batch("train", 0)
    dummy_batch = jax.tree_map(lambda a: jnp.array(a), dummy_batch)

    rng = jax.random.PRNGKey(42)
    model = hk.transform(model_fn)

    with open(os.path.join(model_path, "best"), "rb") as f:
        params = pickle.load(f)

    def compute_logits(params, rng, config, batch):
        _, logits = model.apply(params, rng, config, batch)[:2]
        return logits

    print("getting predictions")
    results = []
    split = "test"
    for dev_index in range(loader.get_number_of_batches(split)):
        raw_batch = loader.get_batch(split, dev_index)
        batch = jax.tree_map(lambda a: jnp.array(a), raw_batch)

        logits = compute_logits(
            femr.models.transformer.convert_params(params, jnp.float16),
            rng,
            config,
            batch,
        )

        logits = np.array(logits)

        p_index = (
            batch["transformer"]["label_indices"] // batch["transformer"]["length"]
        )

        for i in range(batch["num_indices"]):
            logit = logits[i]

            label_pid = raw_batch["patient_ids"][p_index[i]]
            label_age = raw_batch["task"]["label_ages"][i]
            label = raw_batch["task"]["labels"][i]

            offset = raw_batch["offsets"][p_index[i]]
            results.append((label_pid, label_age, offset, label, logit))

    results.sort(key=lambda a: a[:3])
    all_logits = []
    label_pids = []
    label_ages = []
    labels = []
    last_label_idx = None

    for label_pid, label_age, offset, label, logit in results:
        # Ignore duplicate
        if (label_pid, label_age) == last_label_idx:
            continue
        last_label_idx = (label_pid, label_age)
        all_logits.append(logit)
        label_pids.append(label_pid)
        label_ages.append(label_age)
        labels.append(label)

    result = {
        "data_path": path_extract,
        "model": model_path,
        "labels": np.array(labels),
        "predictions": np.array(jax.nn.sigmoid(np.array(all_logits))),
        "patient_ids": np.array(label_pids),
    }

    return result


def run_bootstrap(
    labels: np.array,
    preds: np.array,
    metrics: Dict[str, Callable[[np.array, np.array], float]],
    n_boots: int,
):
    results = {metric + "_bootstrap": [] for metric, _ in metrics.items()}
    n_samples = len(labels)
    np.random.seed(97)
    errs = 0

    for i in range(n_boots):
        ids = np.random.choice(n_samples, n_samples)

        for metric in metrics:
            try:
                results[metric + "_bootstrap"].append(
                    metrics[metric](labels[ids], preds[ids])
                )
            except TypeError:
                results[metric + "_bootstrap"].append(np.nan)
                errs += 1

    print(f"{errs} bootstrap iterations errored out, likely due to insufficient y=1")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adapter model")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_model", type=str)
    parser.add_argument("--correct_calibration", action="store_true")
    parser.add_argument(
        "--model_type",
        type=str,
        default="adapter_model",
        help="adapter_model/linear_probe/clmbr_task_model",
    )

    parser.add_argument(
        "--n_boots", type=int, default=None, help="num bootstrap iterations"
    )

    parser.add_argument("--overwrite", action="store_true")
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

    print(
        f"\n\
    output_dir: {args.path_to_output_dir}\n\
    path_to_model: {args.path_to_model}\n\
    n_boots: {args.n_boots}\n\
    "
    )

    if args.overwrite and os.path.exists(args.path_to_output_dir):
        shutil.rmtree(args.path_to_output_dir, ignore_errors=True)

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

        # load model and metadata
        if args.model_type == "adapter_model":
            results = get_adapter_model_results(args)

        elif args.model_type == "linear_probe":
            results = get_linear_probe_results(args)

        elif args.model_type == "clmbr_task_model":
            results = get_clmbr_task_model_results(args)

        else:
            raise ValueError(
                "`model_type` must be either 'adapter_model' or 'clmbr_task_model' or 'linear_probe'"
            )

        print("evaluating model")
        for metric in metrics:
            results[metric] = metrics[metric](results["labels"], results["predictions"])

        if args.n_boots is not None:
            print(f"evaluating using {args.n_boots} bootstrap iterations")

            boot_results = run_bootstrap(
                results["labels"],
                results["predictions"],
                metrics,
                args.n_boots,
            )

            results = {**results, **boot_results}

        # save
        print("saving results")
        save_to_pkl(results, os.path.join(args.path_to_output_dir, "results.pkl"))

        t_end = int(time.time() - START_TIME)
        print(f"finished in {t_end} seconds")
