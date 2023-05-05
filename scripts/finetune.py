"""
Finetune CLMBR for task.

Supports two-step finetuning proposed in https://arxiv.org/abs/2202.10054:
First step: linear-probing (freeze CLMBR weights, train adapter layer)
Second step: fine-tuning (update CLMBR weights along with adapter layer)
"""

import os
import subprocess
import shutil
import argparse

from src.io import read_msgpack
from src.default_paths import path_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain CLMBR")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--path_to_task_batches", type=str, default=None)
    parser.add_argument("--path_to_og_clmbr_data", type=str, required=True)
    parser.add_argument("--num_batch_threads", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--early_stopping_window_steps", type=int, default=3000)
    parser.add_argument("--two_step_finetuning", default=False, action="store_true")

    args = parser.parse_args()

    PATH_PATIENT_DATABASE = path_extract

    PATH_OG_CLMBR_DATA = args.path_to_og_clmbr_data
    PATH_OG_BATCHES = args.path_to_task_batches
    PATH_OG_MODEL = os.path.join(PATH_OG_CLMBR_DATA, "clmbr_model")

    PATH_OUTPUT_DIR = args.path_to_output_dir
    PATH_BATCHES = os.path.join(PATH_OUTPUT_DIR, "task_batches")
    PATH_MODEL_STEP1 = None
    PATH_MODEL = os.path.join(PATH_OUTPUT_DIR, "clmbr_model")

    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    print(
        f"\n\
    output_dir: {PATH_OUTPUT_DIR}\n\
    original_clmbr_dir: {PATH_OG_CLMBR_DATA}\n\
    task_batches_dir: {PATH_OG_BATCHES}\n\
    path_to_patient_database: {PATH_PATIENT_DATABASE}\n\
    "
    )

    # Copy batches
    if args.overwrite and os.path.exists(PATH_BATCHES):
        print("Copying task batches")
        shutil.rmtree(PATH_BATCHES, ignore_errors=True)

    if not os.path.exists(PATH_BATCHES):
        shutil.copytree(PATH_OG_BATCHES, PATH_BATCHES)

    model_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))[
        "transformer"
    ]

    # Linear probe if two-step is enabled
    if args.two_step_finetuning:
        PATH_MODEL_STEP1 = os.path.join(PATH_OUTPUT_DIR, "clmbr_model_step1")

        if args.overwrite and os.path.exists(PATH_MODEL_STEP1):
            shutil.rmtree(PATH_MODEL_STEP1, ignore_errors=True)

        if not os.path.exists(PATH_MODEL_STEP1):
            cmd = [
                "clmbr_train_model",
                PATH_MODEL_STEP1,
                "--start_from_checkpoint",
                PATH_OG_MODEL,
                "--data_path",
                PATH_PATIENT_DATABASE,
                "--batches_path",
                PATH_BATCHES,
                "--learning_rate",
                str(args.learning_rate),
                "--max_iter",
                str(args.max_iter),
                "--rotary_type",
                model_config["rotary"],
                "--weight_decay",
                str(args.weight_decay),
                "--internal_dropout",
                str(model_config["internal_dropout"]),
                "--n_heads",
                str(model_config["n_heads"]),
                "--n_layers",
                str(model_config["n_layers"]),
                "--attention_width",
                str(model_config["attention_width"]),
                "--num_batch_threads",
                str(args.num_batch_threads),
                "--hidden_size",
                str(model_config["hidden_size"]),
                "--intermediate_size",
                str(model_config["intermediate_size"]),
                "--early_stopping_window_steps",
                str(args.early_stopping_window_steps),
                "--freeze_weights",
            ]

            subprocess.run(cmd)

    # Fine-tune
    if args.overwrite and os.path.exists(PATH_MODEL):
        shutil.rmtree(PATH_MODEL, ignore_errors=True)

    if not os.path.exists(PATH_MODEL):
        if args.two_step_finetuning:
            path_checkpoint = PATH_MODEL_STEP1
        else:
            path_checkpoint = PATH_OG_MODEL

        cmd = [
            "clmbr_train_model",
            PATH_MODEL,
            "--start_from_checkpoint",
            path_checkpoint,
            "--data_path",
            PATH_PATIENT_DATABASE,
            "--batches_path",
            PATH_BATCHES,
            "--learning_rate",
            str(args.learning_rate),
            "--max_iter",
            str(args.max_iter),
            "--rotary_type",
            model_config["rotary"],
            "--weight_decay",
            str(args.weight_decay),
            "--internal_dropout",
            str(model_config["internal_dropout"]),
            "--n_heads",
            str(model_config["n_heads"]),
            "--n_layers",
            str(model_config["n_layers"]),
            "--attention_width",
            str(model_config["attention_width"]),
            "--num_batch_threads",
            str(args.num_batch_threads),
            "--hidden_size",
            str(model_config["hidden_size"]),
            "--intermediate_size",
            str(model_config["intermediate_size"]),
            "--early_stopping_window_steps",
            str(args.early_stopping_window_steps),
        ]

        subprocess.run(cmd)
