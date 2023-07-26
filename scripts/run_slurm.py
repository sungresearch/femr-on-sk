"""
Submits slurm job to run script with config file.
Searches for configs stored under "[root_path]/[configs]/*".
For some, specifying the directory, e.g., "featurize" will run all configs stored there, i.e., "featurize/*.yml".

Usage example:
    python run_slurm.py --label "label.yml"
"""

import argparse
import os
import datetime

from simple_slurm import Slurm
from sklearn.model_selection import ParameterGrid
from src.io import read_yaml
from src.default_paths import path_root
from src.utils import list_dir, get_best_clmbr_model


def get_configs(config_path: str):
    """get list of configs from config_path"""

    if os.path.isdir(config_path):
        configs = [
            read_yaml(os.path.join(config_path, file)) for file in list_dir(config_path)
        ]

    else:
        configs = [read_yaml(config_path)]

    return configs


def setup_slurm(args):
    slurm = Slurm(ignore_pbs=True)

    slurm_args = {}
    for arg, arg_val in vars(args).items():
        if "slurm" in arg and arg_val is not None:
            arg = "_".join(arg.split("_")[1:])

            if "time" in arg:
                slurm_args[arg] = datetime.timedelta(days=arg_val)
                continue

            slurm_args[arg] = arg_val

    slurm.add_arguments(**slurm_args)

    return slurm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--label_sql", type=str, default=None)
    parser.add_argument("--featurize", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--continue_pretrain", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--train_adapter", type=str, default=None)
    parser.add_argument("--train_adapter_few_shots", type=str, default=None)
    parser.add_argument("--evaluate", type=str, default=None)
    parser.add_argument("--slurm_cpus_per_task", type=int, default=4)
    parser.add_argument("--slurm_mem", type=int, default=12000)
    parser.add_argument("--slurm_gres", type=str, default=None)
    parser.add_argument("--slurm_time", type=int, default=1)
    parser.add_argument("--slurm_mail-user", type=str, default=None)
    parser.add_argument("--slurm_mail-type", type=str, default=None)
    parser.add_argument("--slurm_constraint", type=str, default="AlmaLinux8")

    args = parser.parse_args()
    slurm = setup_slurm(args)

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

        PATH_LOGS = os.path.join(path_root, "logs/labeler")
        os.makedirs(PATH_LOGS, exist_ok=True)

        for labeler in config["labelers"]:
            slurm.add_arguments(
                error=os.path.join(PATH_LOGS, f"{labeler}-%J.err"),
                out=os.path.join(PATH_LOGS, f"{labeler}-%J.out"),
                job_name="label",
            )

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

            slurm.sbatch(" ".join(cmd))

    if args.label_sql is not None:
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
            ```
        """
        config = read_yaml(os.path.join(path_root, "configs", args.label_sql))
        PATH_SCRIPT = os.path.join(path_root, "scripts", "label_sql.py")
        PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])

        PATH_LOGS = os.path.join(path_root, "logs/labeler")
        os.makedirs(PATH_LOGS, exist_ok=True)

        for labeler in config["labelers"]:
            slurm.add_arguments(
                error=os.path.join(PATH_LOGS, f"{labeler}-%J.err"),
                out=os.path.join(PATH_LOGS, f"{labeler}-%J.out"),
                job_name="label",
            )

            cmd = [
                "python",
                PATH_SCRIPT,
                os.path.join(PATH_OUTPUT_DIR, f"{labeler}_sql"),
                "--labeler",
                labeler,
            ]

            if config["overwrite"]:
                cmd += ["--overwrite"]

            slurm.sbatch(" ".join(cmd))

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

                # select best CLMBR model if directory contains multiple CLMBR models
                if "clmbr_model" not in list_dir(PATH_CLMBR_DATA):
                    PATH_CLMBR_DATA = os.path.join(
                        path_root,
                        config["featurizer_config"]["path_to_clmbr_data"],
                        get_best_clmbr_model(PATH_CLMBR_DATA),
                    )

            if "labels_to_featurize" in config:
                available_labels = config["labels_to_featurize"]
            else:
                available_labels = list_dir(PATH_LABELS)

            PATH_LOGS = os.path.join(
                path_root, "logs/featurize/", PATH_OUTPUT_DIR.split("/")[-1]
            )

            os.makedirs(PATH_LOGS, exist_ok=True)

            # get labels for which to featurize
            for label in available_labels:
                slurm.add_arguments(
                    error=os.path.join(PATH_LOGS, f"{label}-%J.err"),
                    out=os.path.join(PATH_LOGS, f"{label}-%J.out"),
                    cpus_per_task=12,
                    mem=64000,
                    job_name="featurize",
                )

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

                    if (
                        "is_ontology_expansion" in config["featurizer_config"]
                        and config["featurizer_config"]["is_ontology_expansion"]
                    ):
                        cmd += ["--is_ontology_expansion"]

                elif config["featurizer_config"]["type"] == "clmbr":
                    cmd += ["--clmbr", "--path_to_clmbr_data", PATH_CLMBR_DATA]

                    slurm.add_arguments(gres="gpu:1")

                elif config["featurizer_config"]["type"] == "motor":
                    cmd += ["--motor", "--path_to_clmbr_data", PATH_CLMBR_DATA]

                    slurm.add_arguments(gres="gpu:1")

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                slurm.sbatch(" ".join(cmd))

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
        PATH_CONFIG = os.path.join(path_root, "configs", args.pretrain)
        configs = get_configs(PATH_CONFIG)

        for config in configs:
            PATH_SCRIPT = os.path.join(path_root, "scripts", "pretrain.py")
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])

            PATH_LOGS = os.path.join(
                path_root, "logs/pretrain/", PATH_OUTPUT_DIR.split("/")[-1]
            )

            os.makedirs(PATH_LOGS, exist_ok=True)

            # create hyperparameter grid
            for k, v in config["transformer_config"].items():
                if type(v) == list:
                    continue
                config["transformer_config"][k] = [v]

            model_param_grid = list(ParameterGrid(config["transformer_config"]))

            for params in model_param_grid:
                params_list = []
                for k, v in params.items():
                    if k == "is_hierarchical":
                        continue

                    params_list += ["--" + str(k), str(v)]

                model_name = "CLMBR_" + "_".join(
                    [x.replace("--", "") for x in params_list]
                )

                slurm.add_arguments(
                    error=os.path.join(PATH_LOGS, f"{model_name}-%J.err"),
                    out=os.path.join(PATH_LOGS, f"{model_name}-%J.out"),
                    cpus_per_task=12,
                    mem=64000,
                    gres="gpu:1",
                    job_name="pretrain",
                )

                cmd = [
                    "python",
                    PATH_SCRIPT,
                    os.path.join(PATH_OUTPUT_DIR, model_name),
                ] + params_list

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                if (
                    "is_hierarchical" in config["transformer_config"]
                    and config["transformer_config"]["is_hierarchical"]
                ):
                    cmd += ["--is_hierarchical"]

                slurm.sbatch(" ".join(cmd))

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
        PATH_SCRIPT = os.path.join(path_root, "scripts", "continue_pretrain.py")
        PATH_CONFIG = os.path.join(path_root, "configs", args.featurize)
        configs = get_configs(PATH_CONFIG)

        for config in configs:
            PATH_OUTPUT_DIR = os.path.join(path_root, config["path_to_output_dir"])
            PATH_OG_CLMBR_DATA = os.path.join(
                path_root, config["path_to_og_clmbr_data"]
            )

            # select best CLMBR model if directory contains multiple CLMBR models
            if "clmbr_model" not in list_dir(PATH_OG_CLMBR_DATA):
                PATH_OG_CLMBR_DATA = os.path.join(
                    path_root,
                    config["path_to_og_clmbr_data"],
                    get_best_clmbr_model(PATH_OG_CLMBR_DATA),
                )

            # create hyperparameter grid
            for k, v in config["transformer_config"].items():
                if type(v) == list:
                    continue
                config["transformer_config"][k] = [v]

            model_param_grid = list(ParameterGrid(config["transformer_config"]))

            PATH_LOGS = os.path.join(
                path_root, "logs/continue_pretrain/", PATH_OUTPUT_DIR.split("/")[-1]
            )

            os.makedirs(PATH_LOGS, exist_ok=True)

            for params in model_param_grid:
                params_list = []
                for k, v in params.items():
                    if k == "limit_to_cohort":
                        continue

                    params_list += ["--" + str(k), str(v)]

                model_name = "CLMBR_" + "_".join(
                    [x.replace("--", "") for x in params_list]
                )

                slurm.add_arguments(
                    error=os.path.join(PATH_LOGS, f"{model_name}-%J.err"),
                    out=os.path.join(PATH_LOGS, f"{model_name}-%J.out"),
                    cpus_per_task=12,
                    mem=64000,
                    gres="gpu:1",
                    job_name="dapt",
                )

                cmd = [
                    "python",
                    PATH_SCRIPT,
                    os.path.join(PATH_OUTPUT_DIR, model_name),
                    "--path_to_og_clmbr_data",
                    PATH_OG_CLMBR_DATA,
                ] + params_list

                if "limit_to_cohort" in config["transformer_config"]:
                    cmd += [
                        "--limit_to_cohort",
                        os.path.join(
                            path_root,
                            config["transformer_config"]["limit_to_cohort"][0],
                        ),
                    ]

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                slurm.sbatch(" ".join(cmd))

    if args.finetune is not None:
        """
        fine-tuning of FEMR as proposed in https://arxiv.org/abs/2202.10054

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

            # select best CLMBR model if directory contains multiple CLMBR models
            if "clmbr_model" not in list_dir(PATH_OG_CLMBR_DATA):
                PATH_OG_CLMBR_DATA = os.path.join(
                    path_root,
                    config["path_to_clmbr_data"],
                    get_best_clmbr_model(PATH_OG_CLMBR_DATA),
                )

            available_tasks = list_dir(PATH_LABELS)

            PATH_LOGS = os.path.join(
                path_root, "logs/finetune/", PATH_OUTPUT_DIR.split("/")[-1]
            )

            os.makedirs(PATH_LOGS, exist_ok=True)

            cmds = []
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
                    ]

                    if config["overwrite"]:
                        cmd += ["--overwrite"]

                    cmds.append(" ".join(cmd))

            slurm.add_arguments(
                error=os.path.join(PATH_LOGS, f"{task}_{lr}-%J.err"),
                out=os.path.join(PATH_LOGS, f"{task}_{lr}-%J.out"),
                cpus_per_task=12,
                mem=64000,
                gres="gpu:1",
                job_name="finetune",
            )

            slurm.sbatch("\n".join(cmds))

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

            PATH_LOGS = os.path.join(
                path_root, "logs/train_adapter/", PATH_OUTPUT_DIR.split("/")[-1]
            )

            os.makedirs(PATH_LOGS, exist_ok=True)

            for features in available_features:
                slurm.add_arguments(
                    error=os.path.join(PATH_LOGS, f"{features}-%J.err"),
                    out=os.path.join(PATH_LOGS, f"{features}-%J.out"),
                    cpus_per_task=16,
                    mem=32000,
                    job_name="adapt",
                )

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

                if "scale_features" in config and config["scale_features"]:
                    cmd += ["--scale_features"]

                if "model" in config:
                    cmd += ["--model", config["model"]]

                slurm.sbatch(" ".join(cmd))

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
                PATH_LOGS = os.path.join(
                    path_root,
                    "logs/train_adapter_few_shots/",
                    path_output_dir.split("/")[-1],
                )

                os.makedirs(PATH_LOGS, exist_ok=True)

                for i_iter in range(config["n_iters"]):
                    output_suffix = f"_{train_n}_iter{i_iter}"

                    slurm.add_arguments(
                        error=os.path.join(PATH_LOGS, f"{output_suffix}-%J.err"),
                        out=os.path.join(PATH_LOGS, f"{output_suffix}-%J.out"),
                        cpus_per_task=16,
                        mem=32000,
                        job_name="adapt_fs",
                    )

                    cmds = []
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
                            "--n_jobs",
                            "1",
                        ]

                        if config["overwrite"]:
                            cmd += ["--overwrite"]

                        if "scale_features" in config and config["scale_features"]:
                            cmd += ["--scale_features"]

                        if "model" in config:
                            cmd += ["--model", config["model"]]

                        cmds.append(" ".join(cmd))

                    slurm.sbatch("\n".join(cmds))

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

            PATH_LOGS = os.path.join(path_root, "logs/evaluate")
            os.makedirs(PATH_LOGS, exist_ok=True)

            for model in available_models:
                available_tasks = list_dir(os.path.join(path_models, model))

                slurm.add_arguments(
                    error=os.path.join(PATH_LOGS, f"{model}_%J.err"),
                    out=os.path.join(PATH_LOGS, f"{model}_%J.out"),
                    job_name="evaluate",
                )

                cmds = []
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
                        cmd += ["--n_boots", str(config["n_boots"])]

                    if "model_type" in config:
                        cmd += ["--model_type", config["model_type"]]

                    if config["overwrite"]:
                        cmd += ["--overwrite"]

                    cmds.append(" ".join(cmd))

                slurm.sbatch("\n".join(cmds))