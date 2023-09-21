"""
adapter model stats columns:
    metric (str), m1 (str), m2 (str), m1_performance (str), m2_performance (str), difference (str), reject_h0 (bool)

few shots columns: add train_n
subsample columns: add subsample_size
"""

import os
import argparse

import pandas as pd

from src.io import read_pkl
from src.default_paths import path_root
from src.utils import list_dir
from src.mappings import model_names


def collect_results(
    path_results: str,
    few_shots_experiment: bool = False,
    subsample_experiment: bool = False,
    flip_comparison: bool = False,
) -> pd.DataFrame:
    """
    Stats columns:
        - Metric: str,
        - Model 1 Name: str,
        - Model 2 Name: str,
        - Model 1 Performance: str (median [95% CI]),
        - Model 2 Performance: str (median [95% CI]),
        - Difference: str (median [95% CI]),
        - Reject H0: bool

    If few shots experiment: Add `Num Training Samples`: int
    If subsample experiment: Add `Pretraining Subsample`: float
    """

    df = pd.DataFrame()
    for results in list_dir(path_results):
        try:
            results = read_pkl(os.path.join(path_results, results, "results.pkl"))
        except FileNotFoundError:
            print(f"results.pkl does not exist for {results}. Skipping.")
            continue

        ci = results["results_ci"]
        p = results["results_p"]

        m1 = ci.columns[0]
        m2 = ci.columns[1]

        ci = ci.reset_index().rename(columns={"level_1": "percentile"})
        metrics = ci["metric"].unique()
        for metric in metrics:
            medians = ci.query("metric==@metric and percentile==0.5").round(3)
            lower_cis = ci.query("metric==@metric and percentile==0.025").round(3)
            upper_cis = ci.query("metric==@metric and percentile==0.975").round(3)

            medians = medians.iloc[0].to_dict()
            lower_cis = lower_cis.iloc[0].to_dict()
            upper_cis = upper_cis.iloc[0].to_dict()

            m1_performance = f"{medians[m1]} [{lower_cis[m1]}, {upper_cis[m1]}]"
            m2_performance = f"{medians[m2]} [{lower_cis[m2]}, {upper_cis[m2]}]"
            difference = f"{medians['difference']} [{lower_cis['difference']}, {upper_cis['difference']}]"

            if flip_comparison:
                difference = f"{-medians['difference']} [{-upper_cis['difference']}, {-lower_cis['difference']}]"

            pvalue = p.query("metric==@metric")["p-value"].values[0]

            if subsample_experiment or few_shots_experiment:
                m1_name = "_".join(m1.split("_")[:-1])
                m2_name = "_".join(m2.split("_")[:-1])
                sample_size = m1.split("_")[-1]

            else:
                m1_name = m1
                m2_name = m2

            tmp = pd.DataFrame(
                {
                    "Metric": [metric],
                    "Model 1 Name": model_names[m1_name]
                    if not flip_comparison
                    else model_names[m2_name],
                    "Model 2 Name": model_names[m2_name]
                    if not flip_comparison
                    else model_names[m1_name],
                    "Model 1 Performance": m1_performance
                    if not flip_comparison
                    else m2_performance,
                    "Model 2 Performance": m2_performance
                    if not flip_comparison
                    else m1_performance,
                    "Difference": difference,
                    "P-value": pvalue,
                }
            )

            if subsample_experiment:
                tmp = tmp.assign(**{"Pretraining Subsample": sample_size})

            if few_shots_experiment:
                tmp = tmp.assign(**{"Num Training Samples": sample_size})

            df = pd.concat((df, tmp))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results")
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        default=os.path.join(path_root, "stats"),
    )
    parser.add_argument("--adapter_models", action="store_true")
    parser.add_argument("--adapter_models_few_shots", action="store_true")
    parser.add_argument("--adapter_models_subsample", action="store_true")
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    adapter_models = args.adapter_models or args.all
    adapter_models_few_shots = args.adapter_models_few_shots or args.all
    adapter_models_subsample = args.adapter_models_subsample or args.all

    PATH_OUTPUT_DIR = os.path.join(path_root, "stats")
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    if adapter_models:
        path_results = os.path.join(path_root, "data/stats/adapter_models")
        df = collect_results(path_results, flip_comparison=True)
        df.to_csv(os.path.join(PATH_OUTPUT_DIR, "adapter_models.csv"), index=False)

    if adapter_models_subsample:
        path_results = os.path.join(path_root, "data/stats/adapter_models_subsample")
        df = collect_results(path_results, subsample_experiment=True)
        df.to_csv(
            os.path.join(PATH_OUTPUT_DIR, "adapter_models_subsample.csv"), index=False
        )

    if adapter_models_few_shots:
        path_results = os.path.join(path_root, "data/stats/adapter_models_few_shots")
        df = collect_results(
            path_results, few_shots_experiment=True, flip_comparison=True
        )
        df.to_csv(
            os.path.join(PATH_OUTPUT_DIR, "adapter_models_few_shots.csv"), index=False
        )
