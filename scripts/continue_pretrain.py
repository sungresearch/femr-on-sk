import os
import subprocess
import shutil
import argparse

import pandas as pd

from src.io import read_msgpack
from src.default_paths import path_extract
from src.utils import list_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain CLMBR")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--path_to_og_clmbr_data", type=str, required=True)
    parser.add_argument("--num_batch_threads", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--early_stopping_window_steps", type=int, default=15000)
    parser.add_argument(
        "--limit_to_cohort",
        type=str,
        default=None,
        help="path to labels folder to get patient IDs",
    )
    parser.add_argument(
        "--limit_to_patients_file",
        default=None,
        type=str,
        help="Path to file containing patient_ids to allow in batches",
    )

    args = parser.parse_args()

    PATH_PATIENT_DATABASE = path_extract
    PATH_OG_CLMBR_DATA = args.path_to_og_clmbr_data
    PATH_OG_DICTIONARY = os.path.join(PATH_OG_CLMBR_DATA, "dictionary")
    PATH_OG_MODEL = os.path.join(PATH_OG_CLMBR_DATA, "clmbr_model")

    PATH_OUTPUT_DIR = args.path_to_output_dir
    PATH_DICTIONARY = os.path.join(PATH_OUTPUT_DIR, "dictionary")
    PATH_BATCHES = os.path.join(PATH_OUTPUT_DIR, "clmbr_batches")
    PATH_MODEL = os.path.join(PATH_OUTPUT_DIR, "clmbr_model")

    PATH_LIMIT_PATIENT_IDS = None

    print(
        f"\n\
    output_dir: {PATH_OUTPUT_DIR}\n\
    original_clmbr_dir: {PATH_OG_CLMBR_DATA}\n\
    path_to_patient_database: {PATH_PATIENT_DATABASE}\n\
    "
    )

    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    # RESTRICT CONTINUED PRETRAINING TO PATIENT FILE
    if args.limit_to_cohort is not None:
        PATH_LIMIT_PATIENT_IDS = os.path.join(PATH_OUTPUT_DIR, "cohort_pids")

        if args.overwrite and os.path.exists(PATH_LIMIT_PATIENT_IDS):
            os.remove(PATH_LIMIT_PATIENT_IDS)

        if not os.path.exists(PATH_LIMIT_PATIENT_IDS):
            print(f"limiting pretraining to patient IDs in {args.limit_to_cohort}")
            cohort_pids = []
            labels = list_dir(args.limit_to_cohort)

            for label in labels:
                cohort_pids += pd.read_csv(
                    os.path.join(args.limit_to_cohort, label, "labeled_patients.csv")
                )["patient_id"].tolist()

            cohort_pids = list(set(cohort_pids))

            with open(PATH_LIMIT_PATIENT_IDS, "w") as f:
                for pid in cohort_pids:
                    f.write(f"{pid}\n")

    if args.limit_to_patients_file is not None:
        PATH_LIMIT_PATIENT_IDS = args.limit_to_patients_file

    # Copy dictionary
    if not os.path.exists(PATH_DICTIONARY):
        shutil.copyfile(PATH_OG_DICTIONARY, PATH_DICTIONARY)

    # load model config
    model_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))[
        "transformer"
    ]

    # Create batches
    if args.overwrite and os.path.exists(PATH_BATCHES):
        shutil.rmtree(PATH_BATCHES)

    # We need to grab "transformer_vocab_size" from the original CLMBR
    # loader_config.msgpack. Alternatively, we can get this from model config.msgpack
    clmbr_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))
    vocab_size = clmbr_config["transformer"]["vocab_size"]

    if not os.path.exists(PATH_BATCHES):
        cmd = [
            "clmbr_create_batches",
            PATH_BATCHES,
            "--data_path",
            PATH_PATIENT_DATABASE,
            "--dictionary",
            PATH_DICTIONARY,
            "--transformer_vocab_size",
            str(vocab_size),
            "--task",
            "clmbr",
            "--val_start",
            "70",
            "--test_start",
            "85",
        ]

        # limit to patient file
        if PATH_LIMIT_PATIENT_IDS is not None:
            cmd += [
                "--limit_to_patients_file",
                PATH_LIMIT_PATIENT_IDS,
            ]

        # check is_hierarchical
        if clmbr_config["transformer"]["is_hierarchical"]:
            cmd += ["--is_hierarchical"]

        subprocess.run(cmd)

    # Continue pretraining
    if args.overwrite and os.path.exists(PATH_MODEL):
        shutil.rmtree(PATH_MODEL, ignore_errors=True)

    if not os.path.exists(PATH_MODEL):
        cmd = [
            "clmbr_train_model",
            PATH_MODEL,
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
        ]

        subprocess.run(cmd)
