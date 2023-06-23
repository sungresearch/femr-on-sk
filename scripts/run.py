"""
Runs script with config file.
Searches for configs stored under "[root_path]/[configs]/*".
For some, specifying the directory, e.g., "featurize" will run all configs stored there.

Usage example:
    python run.py --label "label.yml"
"""

import subprocess
import argparse
import os

from sklearn.model_selection import ParameterGrid
from src.io import read_yaml
from src.default_paths import path_root
from src.utils import list_dir


def get_configs(config_path: str):
    """get list of configs from config_path"""

    if os.path.isdir(config_path):
        configs = [
            read_yaml(os.path.join(config_path, file)) for file in list_dir(config_path)
        ]

    else:
        configs = [read_yaml(config_path)]

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--featurize", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--continue_pretrain", type=str, default=None)
    parser.add_argument("--linear_probe", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--train_adapter", type=str, default=None)
    parser.add_argument("--train_adapter_few_shots", type=str, default=None)
    parser.add_argument("--evaluate", type=str, default=None)

    args = parser.parse_args()

    if args.label is not None:
        """
        Run labeler using config file.
        Supports running multiple labelers.

        Example yml config:
            ```
            labelers:
                - mortality
                - long_los
                - readmission
                - thrombocytopenia_lab
                - hyperkalemia_lab
                - hypoglycemia_lab
                - hyponatremia_lab
                - anemia_lab
            path_to_output_dir: data/labels
            overwrite: True
            num_threads: 4
            max_labels_per_patient: 1
            ```
        """
        config = read_yaml(os.path.join(path_root, "configs", args.label))
        PATH_SCRIPT = os.path.join(path_root, "scripts", "label.py")
        PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])

        for labeler in config["labelers"]:
            cmd = [
                "python",
                PATH_SCRIPT,
                os.path.join(PATH_OUTPUT_DIR, labeler),
                "--num_threads",
                str(config["num_threads"]),
                "--max_labels_per_patient",
                str(config["max_labels_per_patient"]),
                "--labeler",
                labeler,
            ]

            if config["overwrite"]:
                cmd += ["--overwrite"]

            subprocess.run(cmd)

    if args.featurize is not None:
        """
        Run featurizer using config file.
        Featurizer will compute features for each label in path_to_labels.
        If args.featurize is a path, will run all config files.

        Example yml config for count featurizer:
            ```
            path_to_labels: data/labels
            path_to_output_dir: data/features/count_sk
            overwrite: True

            featurizer_config:
                type: count
                num_threads: 4
            ```

        Example yml config for CLMBR featurizer:
            ```
            path_to_labels: data/labels
            path_to_output_dir: data/features/clmbr_stanford
            overwrite: True

            featurizer_config:
                type: clmbr
                path_to_clmbr_data: data/clmbr_models/clmbr_stanford
            ```
        """
        PATH_CONFIG = os.path.join(path_root, "configs", args.featurize)
        configs = get_configs(PATH_CONFIG)

        PATH_SCRIPT = os.path.join(path_root, "scripts", "featurize.py")

        NUM_THREADS = 0
        PATH_CLMBR_DATA = None

        for config in configs:
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])
            PATH_LABELS = os.path.join(path_root, config["path_to_labels"])

            if "num_threads" in config["featurizer_config"]:
                NUM_THREADS = config["featurizer_config"]["num_threads"]

            if "path_to_clmbr_data" in config["featurizer_config"]:
                PATH_CLMBR_DATA = os.path.join(
                    path_root, config["featurizer_config"]["path_to_clmbr_data"]
                )

            available_labels = list_dir(PATH_LABELS)
            # get labels for which to featurize
            for label in available_labels:
                cmd = [
                    "python",
                    PATH_SCRIPT,
                    os.path.join(PATH_OUTPUT_DIR, label),
                    "--path_to_labels",
                    os.path.join(PATH_LABELS, label),
                ]

                if config["featurizer_config"]["type"] == "count":
                    cmd += [
                        "--count",
                        "--num_threads",
                        str(NUM_THREADS),
                    ]

                elif config["featurizer_config"]["type"] == "clmbr":
                    cmd += ["--clmbr", "--path_to_clmbr_data", PATH_CLMBR_DATA]

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                subprocess.run(cmd)

    if args.pretrain is not None:
        """
        Pretrain CLMBR using config file.
        Supports multiple hyperparameter settings specified in transformer_config

        Example yml config:
            ```
            path_to_output_dir: data/clmbr_models/clmbr_sk
            overwrite: True
            transformer_config:
                learning_rate:
                    - 1e-5
                    - 1e-4
                rotary_type: global
                n_heads: 12
                n_layers: 12
                max_iter: 1000000
            ```
        """

        config = read_yaml(os.path.join(path_root, "configs", args.pretrain))
        PATH_SCRIPT = os.path.join(path_root, "scripts", "pretrain.py")
        PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])

        # create hyperparameter grid
        for k, v in config["transformer_config"].items():
            if type(v) == list:
                continue
            config["transformer_config"][k] = [v]

        model_param_grid = list(ParameterGrid(config["transformer_config"]))

        for params in model_param_grid:
            params_list = []
            for k, v in params.items():
                params_list += ["--" + str(k), str(v)]

            model_name = "CLMBR_" + "_".join([x.replace("--", "") for x in params_list])

            cmd = [
                "python",
                PATH_SCRIPT,
                os.path.join(PATH_OUTPUT_DIR, model_name),
            ] + params_list

            if config["overwrite"]:
                cmd += ["--overwrite"]

            subprocess.run(cmd)

    if args.continue_pretrain is not None:
        """
        Continued pretraining of CLMBR using config file.
        Supports multiple hyperparameter settings specified in transformer_config

        Example yml config:
            ```
            path_to_output_dir: data/clmbr_models/clmbr_stanford_cp
            path_to_og_clmbr_data: data/clmbr_models/clmbr_stanford
            overwrite: True
            transformer_config:
                learning_rate:
                    - 1e-5
                    - 1e-4
                max_iter: 1000000
            ```
        """
        config = read_yaml(os.path.join(path_root, "configs", args.continue_pretrain))

        PATH_SCRIPT = os.path.join(path_root, "scripts", "continue_pretrain.py")
        PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])
        PATH_OG_CLMBR_DATA = os.path.join(path_root, config["path_to_og_clmbr_data"])

        # create hyperparameter grid
        for k, v in config["transformer_config"].items():
            if type(v) == list:
                continue
            config["transformer_config"][k] = [v]

        model_param_grid = list(ParameterGrid(config["transformer_config"]))

        for params in model_param_grid:
            params_list = []
            for k, v in params.items():
                params_list += ["--" + str(k), str(v)]

            model_name = "CLMBR_" + "_".join([x.replace("--", "") for x in params_list])

            cmd = [
                "python",
                PATH_SCRIPT,
                os.path.join(PATH_OUTPUT_DIR, model_name),
                "--path_to_og_clmbr_data",
                PATH_OG_CLMBR_DATA,
            ] + params_list

            if config["overwrite"]:
                cmd += ["--overwrite"]

            subprocess.run(cmd)

    if args.linear_probe is not None:
        """
        trains a linear probe on top of a frozen CLMBR

        Will train linear probe for each task defined in path_to_labels,
        and will run all config files in the config folder
        """

        PATH_CONFIG = os.path.join(path_root, "configs", args.linear_probe)
        configs = get_configs(PATH_CONFIG)

        for config in configs:
            PATH_SCRIPT = os.path.join(path_root, "scripts", "linear_probe.py")
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])
            PATH_OG_CLMBR_DATA = os.path.join(path_root, config["path_to_clmbr_data"])
            PATH_LABELS = os.path.join(path_root, config["path_to_labels"])

            available_tasks = list_dir(PATH_LABELS)

            for task in available_tasks:
                cmd = [
                    "python",
                    PATH_SCRIPT,
                    os.path.join(PATH_OUTPUT_DIR, task),
                    "--path_to_labels",
                    os.path.join(PATH_LABELS, task),
                    "--path_to_clmbr_data",
                    PATH_OG_CLMBR_DATA,
                ]

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                subprocess.run(cmd)

    if args.finetune is not None:
        """
        fine-tuning of CLMBR as proposed in https://arxiv.org/abs/2202.10054

        Important: the finetuning step assumes that you have finished the linear_probing
        step.

        Will train multiple fine-tuned models if path_to_task_batches contain multiple
        tasks.
        """
        PATH_CONFIG = os.path.join(path_root, "configs", args.finetune)
        configs = get_configs(PATH_CONFIG)

        for config in configs:
            PATH_SCRIPT = os.path.join(path_root, "scripts", "finetune.py")
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])
            PATH_OG_CLMBR_DATA = os.path.join(path_root, config["path_to_clmbr_data"])
            PATH_LABELS = os.path.join(path_root, config["path_to_labels"])

            available_tasks = list_dir(PATH_LABELS)

            for task in available_tasks:
                for lr in config["transformer_config"]["learning_rate"]:
                    cmd = [
                        "python",
                        PATH_SCRIPT,
                        os.path.join(PATH_OUTPUT_DIR, task, str(lr)),
                        "--path_to_labels",
                        os.path.join(PATH_LABELS, task),
                        "--path_to_clmbr_data",
                        PATH_OG_CLMBR_DATA,
                        "--learning_rate",
                        str(lr),
                        "--max_iter",
                        str(config["transformer_config"]["max_iter"]),
                        "--path_to_linear_probe",
                        os.path.join(path_root, config["path_to_linear_probe"], task),
                    ]

                    if config["overwrite"]:
                        cmd += ["--overwrite"]

                    subprocess.run(cmd)

    if args.train_adapter is not None:
        """
        Train adapter model (linear probe) using config file.

        Will train multiple adapter models if path_to_labels & path_to_features
        contain multiple tasks.

        If args.train_adapter is a path, runs all config files.

        Example yml config:
            ```
            path_to_output_dir: data/adapter_models/count_sk
            path_to_labels: data/labels
            path_to_features: data/features/count_sk
            feature_type: count
            overwrite: True
            ```
        """
        PATH_CONFIG = os.path.join(path_root, "configs", args.train_adapter)
        configs = get_configs(PATH_CONFIG)

        for config in configs:
            PATH_SCRIPT = os.path.join(path_root, "scripts", "train_adapter.py")
            PATH_FEATURES = os.path.join(path_root, config["path_to_features"])
            PATH_LABELS = os.path.join(path_root, config["path_to_labels"])
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])

            available_features = list_dir(PATH_FEATURES)

            for features in available_features:
                cmd = [
                    "python",
                    PATH_SCRIPT,
                    os.path.join(PATH_OUTPUT_DIR, features),
                    "--path_to_labels",
                    os.path.join(PATH_LABELS, features),
                    "--path_to_features",
                    os.path.join(PATH_FEATURES, features),
                    "--feature_type",
                    config["feature_type"],
                ]

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                subprocess.run(cmd)

    if args.train_adapter_few_shots is not None:
        """
        Train adapter model with reduced samples using config file
        If path is specified, run using all config files.

        Example yml config:
            ```
            path_to_output_dir: data/adapter_models_few_shots/count_sk
            path_to_labels: data/labels
            path_to_features: data/features/count_sk
            n_iters: 10
            train_n:
                - 2
                - 4
                - 8
                - 16
                - 32
                - 64
                - 128
            feature_type: count
            overwrite: True
            ```
        """
        PATH_CONFIG = os.path.join(path_root, "configs", args.train_adapter_few_shots)
        configs = get_configs(PATH_CONFIG)

        PATH_SCRIPT = os.path.join(path_root, "scripts", "train_adapter.py")

        for config in configs:
            path_features = os.path.join(path_root, config["path_to_features"])
            path_labels = os.path.join(path_root, config["path_to_labels"])

            if config["path_to_output_dir"][-1] == "/":
                config["path_to_output_dir"] = config["path_to_output_dir"][:-1]

            path_output_dir = os.path.join(path_root, config["path_to_output_dir"])
            available_features = list_dir(path_features)

            for train_n in config["train_n"]:
                for i_iter in range(config["n_iters"]):
                    output_suffix = f"_{train_n}_iter{i_iter}"
                    for features in available_features:
                        cmd = [
                            "python",
                            PATH_SCRIPT,
                            os.path.join(path_output_dir + output_suffix, features),
                            "--path_to_labels",
                            os.path.join(path_labels, features),
                            "--path_to_features",
                            os.path.join(path_features, features),
                            "--feature_type",
                            config["feature_type"],
                            "--train_n",
                            str(train_n),
                        ]

                        if config["overwrite"]:
                            cmd += ["--overwrite"]

                        subprocess.run(cmd)

    if args.evaluate is not None:
        """
        Evaluate adapter model using config file
        If args.evaluate_adapter is path, runs all config files.

        Example yml config:
            ```
            path_to_output_dir: data/evaluate/adapter_models
            path_to_models: data/adapter_models
            overwrite: True
            ```
        """

        config_path = os.path.join(path_root, "configs", args.evaluate)
        configs = get_configs(config_path)

        PATH_SCRIPT = os.path.join(path_root, "scripts", "evaluate.py")

        for config in configs:
            path_models = os.path.join(path_root, config["path_to_models"])
            available_models = list_dir(path_models)

            for model in available_models:
                available_tasks = list_dir(os.path.join(path_models, model))

                for task in available_tasks:
                    path_model = os.path.join(path_models, model, task)

                    path_output_dir = os.path.join(
                        path_root, config["path_to_output_dir"], model, task
                    )

                    cmd = [
                        "python",
                        PATH_SCRIPT,
                        path_output_dir,
                        "--path_to_model",
                        path_model,
                    ]

                    if "n_boots" in config:
                        cmd += ["--n_boots", config["n_boots"]]

                    if "model_type" in config:
                        cmd += ["--model_type", config["model_type"]]

                    if config["overwrite"]:
                        cmd += ["--overwrite"]

                    subprocess.run(cmd)
