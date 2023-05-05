import argparse
import os
import shutil
import time
import random
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import msgpack
import numpy as np

import femr.datasets
import femr.extension.dataloader
import femr.models.dataloader
import femr.models.transformer

from typing import Dict, Callable
from sklearn.metrics import roc_auc_score, average_precision_score

from src.io import save_to_pkl, read_pkl, read_features
from src.default_paths import path_extract

"""
TODO: add expected calibration error
"""

metrics = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
}


def get_logistic_regression_results(args):
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


def get_clmbr_task_model_results(args):
    model_dir = os.path.join(args.path_to_model, "clmbr_model")
    batches_path = os.path.join(args.path_to_model, "task_batches")

    print("loading model and batches")
    with open(os.path.join(model_dir, "config.msgpack"), "rb") as f:
        config = msgpack.load(f, use_list=False)

    random.seed(config["seed"])

    config = hk.data_structures.to_immutable_dict(config)
    batch_info_path = os.path.join(batches_path, "batch_info.msgpack")

    loader = femr.extension.dataloader.BatchLoader(path_extract, batch_info_path)

    def model_fn(config, batch):
        model = femr.models.transformer.EHRTransformer(config)(batch, is_training=False)
        return model

    dummy_batch = loader.get_batch("train", 0)
    dummy_batch = jax.tree_map(lambda a: jnp.array(a), dummy_batch)

    rng = jax.random.PRNGKey(42)
    model = hk.transform(model_fn)

    with open(os.path.join(model_dir, "best"), "rb") as f:
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
        "model": model_dir,
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

    for i in range(n_boots):
        ids = np.random.choice(n_samples, n_samples)

        for metric in metrics:
            results[metric + "_bootstrap"].append(
                metrics[metric](labels[ids], preds[ids])
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adapter model")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_model", type=str)

    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic_regression",
        help="logistic_regression/clmbr_task_model",
    )

    parser.add_argument(
        "--n_boots", type=int, default=None, help="num bootstrap iterations"
    )

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
        if args.model_type == "logistic_regression":
            results = get_logistic_regression_results(args)

        elif args.model_type == "clmbr_task_model":
            results = get_clmbr_task_model_results(args)

        else:
            raise ValueError(
                "`model_type` must be either 'logistic_regression' or 'clmbr_task_model'"
            )

        print("evaluating model")
        for metric in metrics:
            results[metric] = metrics[metric](results["labels"], results["predictions"])

        if args.n_boots is not None:
            print(f"evaluating using {args.n_boot} bootstrap iterations")
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
