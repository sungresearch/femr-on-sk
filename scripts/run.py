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
import pdb

from itertools import zip_longest
from sklearn.model_selection import ParameterGrid
from src.utils import read_yaml
from src.default_paths import path_root


def list_dir(path: str):
    """get list of file/directory names excluding nb checkpoints"""
    
    return [
        x for x in os.listdir(path)
        if x != ".ipynb_checkpoints"
    ]


def get_configs(config_path: str):
    """get list of configs from config_path"""
    
    if os.path.isdir(config_path):
        configs = [
            read_yaml(
                os.path.join(
                    config_path, file
                )
            )
            for file in list_dir(config_path)
        ]

    else:
        configs = [read_yaml(config_path)]
    
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--featurize", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--train_adapter", type=str, default=None)
    parser.add_argument("--evaluate_adapter", type=str, default=None)
    
    parser.add_argument(
        "--train_adapter_few_shots", 
        type=str, default=None, 
        help="train for few shots experiment"
    )
    
    args = parser.parse_args()
    
    if args.label is not None:
        """
        Run labeler using config file
        """
        config = read_yaml(
            os.path.join(path_root, "configs", args.label)
        )
        
        path_script = os.path.join(path_root, "scripts", "label.py")
        
        for labeler in config["labelers"]:
            cmd = [
                "python", path_script,
                os.path.join(config["path_to_output_dir"], labeler),
                "--num_threads", str(config["num_threads"]),
                "--max_labels_per_patient", str(config["max_labels_per_patient"]),
                "--labeler", labeler,
            ]

            if config["overwrite"]:
                cmd += ["--overwrite"]

            subprocess.run(cmd)
            
            
    if args.featurize is not None:
        """
        Run featurizer using config file. 
        If path is specified, run using all config files. 
        Config file specifies whether to run count or CLMBR featurizer.
        """  
        config_path = os.path.join(
            path_root, "configs", args.featurize
        )
        
        configs = get_configs(config_path)
        path_script = os.path.join(path_root, "scripts", "featurize.py")
        
        for config in configs:
            available_labels = list_dir(config["path_to_labels"])
            # get labels for which to featurize
            for label in available_labels:
                
                cmd = [
                    "python", path_script,
                    os.path.join(config["path_to_output_dir"], label),
                    "--path_to_labels", os.path.join(config["path_to_labels"], label),
                ]

                if config["featurizer_config"]["type"] == "count":
                    cmd += [
                        "--count",
                        "--num_threads", str(config["featurizer_config"]["num_threads"]),
                    ]
                    
                    if "ontology_expansion" in config["featurizer_config"] and config["featurizer_config"]["ontology_expansion"]:
                        cmd += ["--ontology_expansion"]
                    
                elif config["featurizer_config"]["type"] == "clmbr":
                    cmd += [
                        "--clmbr",
                        "--path_to_clmbr_data", config["featurizer_config"]["path_to_clmbr_data"]
                    ]

                if config["overwrite"]:                    
                    cmd += ["--overwrite"]
                    
                subprocess.run(cmd)
            
            
    if args.pretrain is not None:
        """
        Pretrain using config file.
        """
        
        config = read_yaml(
            os.path.join(path_root, "configs", args.pretrain)
        )
        
        path_script = os.path.join(path_root, "scripts", "pretrain.py")
        
        # convert items to lists
        for k,v in config['transformer_config'].items():
            if type(v) == list:
                continue
            config['transformer_config'][k] = [v]
        
        model_param_grid = list(ParameterGrid(config['transformer_config']))

        for params in model_param_grid:
            params_list = []
            for k,v in params.items():
                params_list+=["--" + str(k), str(v)]
                
            model_name = "CLMBR_" + "_".join(
                [x.replace("--" , "") for x in params_list]
            )
            
            cmd = [
                "python", path_script,
                os.path.join(config["path_to_output_dir"], model_name),
            ] + params_list
            
            if config["overwrite"]:
                cmd += ["--overwrite"]
            
            subprocess.run(cmd)
            
            
    if args.finetune is not None:
        """
        Finetune pretrained CLMBR using config file.
        """
        
        config = read_yaml(
            os.path.join(path_root, "configs", args.finetune)
        )
        
        path_script = os.path.join(path_root, "scripts", "finetune.py")
        
        # convert items to lists
        for k,v in config['transformer_config'].items():
            if type(v) == list:
                continue
            config['transformer_config'][k] = [v]
            
        model_param_grid = list(ParameterGrid(config['transformer_config']))

        for params in model_param_grid:
            params_list = []
            for k,v in params.items():
                params_list+=["--" + str(k), str(v)]
                
            model_name = "CLMBR_" + "_".join(
                [x.replace("--" , "") for x in params_list]
            )
            
            cmd = [
                "python", path_script,
                os.path.join(config["path_to_output_dir"], model_name),
                "--path_to_og_clmbr_data", config["path_to_og_clmbr_data"],
            ] + params_list
            
            if config["overwrite"]:
                cmd += ["--overwrite"]
                
            if config["finetune_last_layer"]:
                cmd += ["--finetune_last_layer"]
                
            subprocess.run(cmd)
            
            
    if args.train_adapter is not None:
        """
        Train adapter model using config file
        If path is specified, run using all config files. 
        """
        config_path = os.path.join(
            path_root, "configs", args.train_adapter
        )
        
        configs = get_configs(config_path)
        path_script = os.path.join(path_root, "scripts", "train_adapter.py")
        
        for config in configs:
            available_features = list_dir(config["path_to_features"])
            for features in available_features:

                cmd = [
                    "python", path_script, 
                    os.path.join(config["path_to_output_dir"], features),
                    "--path_to_labels", os.path.join(config["path_to_labels"], features),
                    "--path_to_features", os.path.join(config["path_to_features"], features),
                    "--feature_type", config["feature_type"]
                ]

                if config["overwrite"]:
                    cmd += ["--overwrite"]

                subprocess.run(cmd)
   

    if args.train_adapter_few_shots is not None:
        """
        Train adapter model with reduced samples using config file
        If path is specified, run using all config files. 
        """
        config_path = os.path.join(
            path_root, "configs", args.train_adapter_few_shots
        )
        
        configs = get_configs(config_path)
        path_script = os.path.join(path_root, "scripts", "train_adapter.py")
        
        for config in configs:
            available_features = list_dir(config["path_to_features"])
            for train_n in config['train_n']:
                
                if config["path_to_output_dir"][-1] == "/":
                    config["path_to_output_dir"] = config["path_to_output_dir"][:-1]
                    
                for i_iter in range(config["n_iters"]):
                    path_to_output_dir = config["path_to_output_dir"] + f"_{train_n}_iter{i_iter}"

                    for features in available_features:

                        cmd = [
                            "python", path_script, 
                            os.path.join(path_to_output_dir, features),
                            "--path_to_labels", os.path.join(config["path_to_labels"], features),
                            "--path_to_features", os.path.join(config["path_to_features"], features),
                            "--feature_type", config["feature_type"],
                            "--train_n", str(train_n),
                        ]

                        if config["overwrite"]:
                            cmd += ["--overwrite"]

                        subprocess.run(cmd)
                    
                    
    if args.evaluate_adapter is not None:
        """
        Evaluate adapter model using config file
        If path is specified, run using all config files. 
        """
        
        config_path = os.path.join(
            path_root, "configs", args.evaluate_adapter
        )
        
        configs = get_configs(config_path)
        path_script = os.path.join(path_root, "scripts", "evaluate_adapter.py")
        
        for config in configs:
            path_to_models = config["path_to_models"]
            available_models = list_dir(path_to_models)
            
            for model in available_models:
                available_tasks = list_dir(os.path.join(path_to_models, model))
                
                for task in available_tasks:
                    path_to_model = os.path.join(
                        path_to_models, model, task
                    )
                    
                    path_to_output_dir = os.path.join(
                        config["path_to_output_dir"], model, task
                    )
                    
                    cmd = [
                        "python", path_script,
                        path_to_output_dir,
                        "--path_to_model", path_to_model,
                    ]
                    
                    if "n_boots" in config:
                        cmd += ["--n_boots", config["n_boots"]]
                    
                    if config["overwrite"]:
                        cmd += ["--overwrite"]
                    
                    subprocess.run(cmd)