"""
Finetune CLMBR for task as proposed in https://arxiv.org/abs/2202.10054
Step 1: train linear probe
Step 2: end-to-end fine-tuning including the linear probe
"""

import os
import subprocess
import shutil
import argparse

from src.io import read_msgpack
from src.default_paths import path_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune FEMR")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_labels", type=str, required=True)
    parser.add_argument("--path_to_clmbr_data", type=str, required=True)
    parser.add_argument("--num_batch_threads", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--early_stopping_window_steps", type=int, default=2000)
    parser.add_argument("--overwrite", default=False, action="store_true")

    args = parser.parse_args()

    PATH_PATIENT_DATABASE = path_extract
    PATH_OG_CLMBR_DATA = args.path_to_clmbr_data
    PATH_OG_MODEL = os.path.join(PATH_OG_CLMBR_DATA, "clmbr_model")
    PATH_DICTIONARY = os.path.join(PATH_OG_CLMBR_DATA, "dictionary")
    PATH_SURVIVAL_DICTIONARY = None

    if os.path.exists(os.path.join(PATH_OG_CLMBR_DATA, "survival_dictionary")):
        PATH_SURVIVAL_DICTIONARY = os.path.join(
            PATH_OG_CLMBR_DATA, "survival_dictionary"
        )

    PATH_LABELS = args.path_to_labels

    PATH_OUTPUT_DIR = args.path_to_output_dir
    PATH_BATCHES = os.path.join(PATH_OUTPUT_DIR, "task_batches")
    PATH_NEW_CLMBR_MODEL = os.path.join(PATH_OUTPUT_DIR, "clmbr_model")
    PATH_LINEAR_PROBE = os.path.join(PATH_OUTPUT_DIR, "linear_probe")

    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    print(
        f"\n\
    task-specific finetuning of foundation model\n\
    output_dir: {PATH_OUTPUT_DIR}\n\
    original_clmbr_dir: {PATH_OG_CLMBR_DATA}\n\
    path_to_patient_database: {PATH_PATIENT_DATABASE}\n\
    path_to_labels: {PATH_LABELS}\
    "
    )

    # Create task batches
    if args.overwrite and os.path.exists(PATH_BATCHES):
        shutil.rmtree(PATH_BATCHES, ignore_errors=True)

    model_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))[
        "transformer"
    ]

    if not os.path.exists(PATH_BATCHES):
        vocab_size = model_config["vocab_size"]

        cmd = [
            "clmbr_create_batches",
            PATH_BATCHES,
            "--data_path",
            PATH_PATIENT_DATABASE,
            "--dictionary",
            PATH_DICTIONARY,
            "--task",
            "labeled_patients",
            "--labeled_patients_path",
            os.path.join(PATH_LABELS, "labeled_patients.csv"),
            "--transformer_vocab_size",
            str(vocab_size),
            "--val_start",
            "70",
            "--test_start",
            "85",
        ]

        if PATH_SURVIVAL_DICTIONARY is not None:
            cmd += ["--clmbr_survival_dictionary_path", PATH_SURVIVAL_DICTIONARY]

        if model_config["transformer"]["is_hierarchical"]:
            cmd += ["--is_hierarchical"]

        subprocess.run(cmd)

    # Train linear-probe
    if args.overwrite and os.path.exists(PATH_LINEAR_PROBE):
        shutil.rmtree(PATH_LINEAR_PROBE, ignore_errors=True)

    if not os.path.exists(PATH_LINEAR_PROBE):
        cmd = [
            "clmbr_train_linear_probe",
            PATH_LINEAR_PROBE,
            "--data_path",
            PATH_PATIENT_DATABASE,
            "--batches_path",
            PATH_BATCHES,
            "--model_dir",
            PATH_OG_MODEL,
        ]

        subprocess.run(cmd)

    # Fine-tune
    if args.overwrite and os.path.exists(PATH_NEW_CLMBR_MODEL):
        shutil.rmtree(PATH_NEW_CLMBR_MODEL, ignore_errors=True)

    if not os.path.exists(PATH_NEW_CLMBR_MODEL):
        cmd = [
            "clmbr_train_model",
            PATH_NEW_CLMBR_MODEL,
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
            "--linear_probe",
            str(PATH_LINEAR_PROBE),
        ]

        subprocess.run(cmd)
