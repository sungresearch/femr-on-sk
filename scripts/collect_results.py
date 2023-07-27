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
    parser.add_argument("--no_adapter_models", action="store_true")
    parser.add_argument("--no_adapter_models_few_shots", action="store_true")
    parser.add_argument("--no_clmbr_finetuned", action="store_true")
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    adapter_models = not args.no_adapter_models
    adapter_models_few_shots = not args.no_adapter_models_few_shots
    clmbr_finetuned = not args.no_clmbr_finetuned

    # collect adapter models results
    if adapter_models:
        df = pd.DataFrame()

        PATH_RESULTS = os.path.join(path_root, "data/evaluate/adapter_models")
        PATH_OUT = os.path.join(args.path_to_output_dir, "adapter_models")
        os.makedirs(PATH_OUT, exist_ok=True)

        for model in list_dir(PATH_RESULTS):
            path_model = os.path.join(PATH_RESULTS, model)

            for task in list_dir(path_model):
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

            for task in list_dir(path_model):
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
                                "nboot": len(results["auroc_bootstrap"]),
                                "path_results": path_task,
                            }
                        ),
                    )
                )

        df.to_csv(os.path.join(PATH_OUT, "results.csv"), index=False)
