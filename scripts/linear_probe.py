"""
Trains linear probe over patient representations [CLMBR / Count]
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
    parser.add_argument("--path_to_labels", type=str, required=True)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--path_to_clmbr_data", type=str, required=True)

    args = parser.parse_args()

    PATH_PATIENT_DATABASE = path_extract
    PATH_CLMBR_DATA = args.path_to_clmbr_data
    PATH_MODEL = os.path.join(PATH_CLMBR_DATA, "clmbr_model")
    PATH_DICTIONARY = os.path.join(PATH_CLMBR_DATA, "dictionary")
    PATH_LABELS = args.path_to_labels

    PATH_OUTPUT_DIR = args.path_to_output_dir
    PATH_BATCHES = os.path.join(PATH_OUTPUT_DIR, "task_batches")
    PATH_LINEAR_PROBE = os.path.join(PATH_OUTPUT_DIR, "model")

    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    print(
        f"\n\
    linear probe\n\
    output_dir: {PATH_OUTPUT_DIR}\n\
    clmbr_dir: {PATH_CLMBR_DATA}\n\
    path_to_patient_database: {PATH_PATIENT_DATABASE}\n\
    path_to_labels: {PATH_LABELS}\
    "
    )

    # Create task batches
    if args.overwrite and os.path.exists(PATH_BATCHES):
        shutil.rmtree(PATH_BATCHES, ignore_errors=True)

    model_config = read_msgpack(os.path.join(PATH_MODEL, "config.msgpack"))[
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

        subprocess.run(cmd)

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
            PATH_MODEL,
        ]

        subprocess.run(cmd)
