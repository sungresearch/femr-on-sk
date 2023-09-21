import os
import argparse

import numpy as np
import pandas as pd

from src.io import read_pkl
from src.default_paths import path_root
from src.utils import list_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results")
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        default=os.path.join(path_root, "results", "raw"),
    )
    parser.add_argument("--adapter_models", action="store_true")
    parser.add_argument("--adapter_models_few_shots", action="store_true")
    parser.add_argument("--adapter_models_subsample", action="store_true")
    parser.add_argument("--clmbr_finetuned", action="store_true")
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    adapter_models = args.adapter_models or args.all
    adapter_models_few_shots = args.adapter_models_few_shots or args.all
    adapter_models_subsample = args.adapter_models_subsample or args.all
    clmbr_finetuned = args.clmbr_finetuned or args.all

    tasks = [
        "mortality_sql",
        "long_los_sql",
        "readmission_sql",
        "anemia_lab_sql",
        "hyperkalemia_lab_sql",
        "hypoglycemia_lab_sql",
        "hyponatremia_lab_sql",
        "thrombocytopenia_lab_sql",
    ]

    # collect adapter models results
    if adapter_models:
        df = pd.DataFrame()

        PATH_RESULTS = os.path.join(path_root, "data/evaluate/adapter_models")
        PATH_OUT = os.path.join(args.path_to_output_dir, "adapter_models")
        os.makedirs(PATH_OUT, exist_ok=True)

        for model in list_dir(PATH_RESULTS):
            path_model = os.path.join(PATH_RESULTS, model)

            for task in tasks:
                if "results.pkl" not in list_dir(os.path.join(path_model, task)):
                    print(f"'results.pkl' not found, skipping {task} for {model}")
                    continue

                path_task = os.path.join(path_model, task, "results.pkl")
                results = read_pkl(path_task)

                df = pd.concat(
                    (
                        df,
                        pd.DataFrame(
                            {
                                "model": [model],
                                "task": task,
                                "auroc": results["auroc"],
                                "auprc": results["auprc"],
                                "auprc_c": results["auprc_c"],
                                "ece": results["ece"],
                                "auroc_lower_ci": np.percentile(
                                    results["auroc_bootstrap"], 2.5
                                ),
                                "auroc_upper_ci": np.percentile(
                                    results["auroc_bootstrap"], 97.5
                                ),
                                "auprc_lower_ci": np.percentile(
                                    results["auprc_bootstrap"], 2.5
                                ),
                                "auprc_upper_ci": np.percentile(
                                    results["auprc_bootstrap"], 97.5
                                ),
                                "auprc_c_lower_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 2.5
                                ),
                                "auprc_c_upper_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 97.5
                                ),
                                "ece_lower_ci": np.percentile(
                                    results["ece_bootstrap"], 2.5
                                ),
                                "ece_upper_ci": np.percentile(
                                    results["ece_bootstrap"], 97.5
                                ),
                                "nboot": len(results["auroc_bootstrap"]),
                                "path_results": path_task,
                            }
                        ),
                    )
                )

        df.to_csv(os.path.join(PATH_OUT, "results.csv"), index=False)

    # collect finetuned models results
    if clmbr_finetuned:
        df = pd.DataFrame()

        PATH_RESULTS = os.path.join(path_root, "data/evaluate/clmbr_finetuned")
        PATH_OUT = os.path.join(args.path_to_output_dir, "clmbr_finetuned")
        os.makedirs(PATH_OUT, exist_ok=True)

        for model in list_dir(PATH_RESULTS):
            path_model = os.path.join(PATH_RESULTS, model)

            for task in tasks:
                if "results.pkl" not in list_dir(os.path.join(path_model, task)):
                    print(
                        f"'results.pkl' not found, skipping {task} for {model} at {path_model}"
                    )
                    continue

                path_task = os.path.join(path_model, task, "results.pkl")
                results = read_pkl(path_task)

                df = pd.concat(
                    (
                        df,
                        pd.DataFrame(
                            {
                                "model": [model],
                                "task": task,
                                "auroc": results["auroc"],
                                "auprc": results["auprc"],
                                "auprc_c": results["auprc_c"],
                                "ece": results["ece"],
                                "auroc_lower_ci": np.percentile(
                                    results["auroc_bootstrap"], 2.5
                                ),
                                "auroc_upper_ci": np.percentile(
                                    results["auroc_bootstrap"], 97.5
                                ),
                                "auprc_lower_ci": np.percentile(
                                    results["auprc_bootstrap"], 2.5
                                ),
                                "auprc_upper_ci": np.percentile(
                                    results["auprc_bootstrap"], 97.5
                                ),
                                "auprc_c_lower_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 2.5
                                ),
                                "auprc_c_upper_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 97.5
                                ),
                                "ece_lower_ci": np.percentile(
                                    results["ece_bootstrap"], 2.5
                                ),
                                "ece_upper_ci": np.percentile(
                                    results["ece_bootstrap"], 97.5
                                ),
                                "nboot": len(results["auroc_bootstrap"]),
                                "path_results": path_task,
                            }
                        ),
                    )
                )

        df.to_csv(os.path.join(PATH_OUT, "results.csv"), index=False)

    if adapter_models_few_shots:
        df = pd.DataFrame()

        PATH_RESULTS = os.path.join(path_root, "data/evaluate/adapter_models_few_shots")
        PATH_OUT = os.path.join(args.path_to_output_dir, "adapter_models_few_shots")
        os.makedirs(PATH_OUT, exist_ok=True)

        for model_dir in list_dir(PATH_RESULTS):
            model = "_".join(model_dir.split("_")[:-2])
            n_shots = model_dir.split("_")[-2]
            iteration = model_dir.split("_")[-1].split("iter")[-1]

            path_model = os.path.join(PATH_RESULTS, model_dir)

            for task in tasks:
                if "results.pkl" not in list_dir(os.path.join(path_model, task)):
                    print(
                        f"'results.pkl' not found, skipping {task} for {model} at {path_model}"
                    )
                    continue

                path_task = os.path.join(path_model, task, "results.pkl")
                results = read_pkl(path_task)

                df = pd.concat(
                    (
                        df,
                        pd.DataFrame(
                            {
                                "model": [model],
                                "n_shots": n_shots,
                                "iteration": iteration,
                                "task": task,
                                "auroc": results["auroc"],
                                "auprc": results["auprc"],
                                "auprc_c": results["auprc_c"],
                                "ece": results["ece"],
                                "auroc_lower_ci": np.percentile(
                                    results["auroc_bootstrap"], 2.5
                                )
                                if "auroc_bootstrap" in results.keys()
                                else np.nan,
                                "auroc_upper_ci": np.percentile(
                                    results["auroc_bootstrap"], 97.5
                                )
                                if "auroc_bootstrap" in results.keys()
                                else np.nan,
                                "auprc_lower_ci": np.percentile(
                                    results["auprc_bootstrap"], 2.5
                                )
                                if "auprc_bootstrap" in results.keys()
                                else np.nan,
                                "auprc_upper_ci": np.percentile(
                                    results["auprc_bootstrap"], 97.5
                                )
                                if "auprc_bootstrap" in results.keys()
                                else np.nan,
                                "auprc_c_lower_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 2.5
                                )
                                if "auprc_c_bootstrap" in results.keys()
                                else np.nan,
                                "auprc_c_upper_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 97.5
                                )
                                if "auprc_c_bootstrap" in results.keys()
                                else np.nan,
                                "ece_lower_ci": np.percentile(
                                    results["ece_bootstrap"], 2.5
                                )
                                if "ece_bootstrap" in results.keys()
                                else np.nan,
                                "ece_upper_ci": np.percentile(
                                    results["ece_bootstrap"], 97.5
                                )
                                if "ece_bootstrap" in results.keys()
                                else np.nan,
                                "nboot": len(results["auroc_bootstrap"])
                                if "auroc_bootstrap" in results.keys()
                                else 0,
                                "path_results": path_task,
                            }
                        ),
                    )
                )

        df.to_csv(os.path.join(PATH_OUT, "results.csv"), index=False)

    if adapter_models_subsample:
        df = pd.DataFrame()

        PATH_RESULTS = os.path.join(path_root, "data/evaluate/adapter_models_subsample")
        PATH_OUT = os.path.join(args.path_to_output_dir, "adapter_models_subsample")
        os.makedirs(PATH_OUT, exist_ok=True)

        for model_dir in list_dir(PATH_RESULTS):
            model = "_".join(model_dir.split("_")[:-1])
            perc_samples = model_dir.split("_")[-1]

            path_model = os.path.join(PATH_RESULTS, model_dir)

            for task in tasks:
                if "results.pkl" not in list_dir(os.path.join(path_model, task)):
                    print(
                        f"'results.pkl' not found, skipping {task} for {model} at {path_model}"
                    )
                    continue

                path_task = os.path.join(path_model, task, "results.pkl")
                results = read_pkl(path_task)

                df = pd.concat(
                    (
                        df,
                        pd.DataFrame(
                            {
                                "model": [model],
                                "perc_samples": perc_samples,
                                "task": task,
                                "auroc": results["auroc"],
                                "auprc": results["auprc"],
                                "auprc_c": results["auprc_c"],
                                "ece": results["ece"],
                                "auroc_lower_ci": np.percentile(
                                    results["auroc_bootstrap"], 2.5
                                ),
                                "auroc_upper_ci": np.percentile(
                                    results["auroc_bootstrap"], 97.5
                                ),
                                "auprc_lower_ci": np.percentile(
                                    results["auprc_bootstrap"], 2.5
                                ),
                                "auprc_upper_ci": np.percentile(
                                    results["auprc_bootstrap"], 97.5
                                ),
                                "auprc_c_lower_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 2.5
                                ),
                                "auprc_c_upper_ci": np.percentile(
                                    results["auprc_c_bootstrap"], 97.5
                                ),
                                "ece_lower_ci": np.percentile(
                                    results["ece_bootstrap"], 2.5
                                ),
                                "ece_upper_ci": np.percentile(
                                    results["ece_bootstrap"], 97.5
                                ),
                                "nboot": len(results["auroc_bootstrap"]),
                                "path_results": path_task,
                            }
                        ),
                    )
                )

        df.to_csv(os.path.join(PATH_OUT, "results.csv"), index=False)
