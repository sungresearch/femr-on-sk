import os
import subprocess
import shutil
import argparse

import numpy as np

from src.io import read_msgpack
from src.default_paths import path_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain CLMBR")
    parser.add_argument(
        "path_to_output_dir",
        type=str,
        help=("Path to save CLMBR data"),
    )
    parser.add_argument("--task", type=str, default="clmbr")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--is_hierarchical", default=False, action="store_true")
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--rotary_type", type=str, required=True)
    parser.add_argument("--num_batch_threads", type=int, default=3)
    parser.add_argument("--start_from_checkpoint", type=str, default=None)
    parser.add_argument("--token_dropout", type=float, default=0)
    parser.add_argument("--internal_dropout", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--early_stopping_window_steps", type=int, default=15000)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Transformer hidden size"
    )
    parser.add_argument(
        "--limit_to_patients_file",
        default=None,
        type=str,
        help="Path to file containing patient_ids to allow in batches",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="Transformer intermediate layer size",
    )
    parser.add_argument(
        "--n_heads", type=int, default=12, help="Transformer # of heads"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="Transformer # of layers"
    )
    parser.add_argument(
        "--attention_width", type=int, default=512, help="Transformer attention width."
    )
    parser.add_argument("--transformer_vocab_size", type=int, default=None)
    parser.add_argument("--clmbr_survival_dim", type=int, default=None)

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = path_extract
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    PATH_DICTIONARY = os.path.join(PATH_TO_OUTPUT_DIR, "dictionary")
    PATH_BATCHES = os.path.join(PATH_TO_OUTPUT_DIR, "clmbr_batches")
    PATH_MODEL = os.path.join(PATH_TO_OUTPUT_DIR, "clmbr_model")

    print(
        f"\n\
    output_dir: {PATH_TO_OUTPUT_DIR}\n\
    path_to_patient_database: {PATH_TO_PATIENT_DATABASE}\n\
    "
    )

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Create CLMBR vocabulary
    if args.overwrite and os.path.exists(PATH_DICTIONARY):
        os.remove(PATH_DICTIONARY)

    if not os.path.exists(PATH_DICTIONARY):
        subprocess.run(
            [
                "clmbr_create_dictionary",
                PATH_DICTIONARY,
                "--data_path",
                PATH_TO_PATIENT_DATABASE,
            ]
        )

    if args.task == "survival_clmbr":
        PATH_SURVIVAL_DICTIONARY = os.path.join(
            PATH_TO_OUTPUT_DIR, "survival_dictionary"
        )

        if args.overwrite and os.path.exists(PATH_SURVIVAL_DICTIONARY):
            os.remove(PATH_SURVIVAL_DICTIONARY)

        if not os.path.exists(PATH_SURVIVAL_DICTIONARY):
            subprocess.run(
                [
                    "clmbr_create_survival_dictionary",
                    PATH_SURVIVAL_DICTIONARY,
                    "--data_path",
                    PATH_TO_PATIENT_DATABASE,
                    "--num_buckets",
                    "8",
                    "--size",
                    "1024",
                ]
            )

    # Create batches
    if args.overwrite and os.path.exists(PATH_BATCHES):
        shutil.rmtree(PATH_BATCHES)

    # Note that transformer vocabulary size can be at most
    # the length of the dictionary entries, and so since
    # at SK we have a relatively small dictionary size, we
    # need to restrict transformer vocab size.
    # Further, we're rounding it down to the nearest power of 2 for optimization
    vocab_size = args.transformer_vocab_size
    if vocab_size is None:
        dictionary = read_msgpack(PATH_DICTIONARY)
        vocab_size = 2 ** int(np.log2(len(dictionary["regular"])))

        if args.is_hierarchical:
            vocab_size = 2 ** int(np.log2(len(dictionary["ontology_rollup"])))

    if not os.path.exists(PATH_BATCHES):
        cmd = [
            "clmbr_create_batches",
            PATH_BATCHES,
            "--data_path",
            PATH_TO_PATIENT_DATABASE,
            "--dictionary",
            PATH_DICTIONARY,
            "--transformer_vocab_size",
            str(vocab_size),
            "--task",
            args.task,
            "--val_start",
            "70",
            "--test_start",
            "85",
        ]

        if args.task == "survival_clmbr":
            cmd += ["--clmbr_survival_dictionary_path", PATH_SURVIVAL_DICTIONARY]

        if args.is_hierarchical:
            cmd += ["--is_hierarchical"]

        # limit to patient file
        if args.limit_to_patients_file is not None:
            cmd += [
                "--limit_to_patients_file",
                args.limit_to_patients_file,
            ]

        subprocess.run(cmd)

    # Pretrain
    if args.overwrite and os.path.exists(PATH_MODEL):
        shutil.rmtree(PATH_MODEL, ignore_errors=True)

    if not os.path.exists(PATH_MODEL):
        cmd = [
            "clmbr_train_model",
            PATH_MODEL,
            "--data_path",
            PATH_TO_PATIENT_DATABASE,
            "--batches_path",
            PATH_BATCHES,
            "--learning_rate",
            str(args.learning_rate),
            "--max_iter",
            str(args.max_iter),
            "--rotary_type",
            args.rotary_type,
            "--weight_decay",
            str(args.weight_decay),
            "--internal_dropout",
            str(args.internal_dropout),
            "--token_dropout",
            str(args.token_dropout),
            "--n_heads",
            str(args.n_heads),
            "--n_layers",
            str(args.n_layers),
            "--attention_width",
            str(args.attention_width),
            "--num_batch_threads",
            str(args.num_batch_threads),
            "--hidden_size",
            str(args.hidden_size),
            "--intermediate_size",
            str(args.intermediate_size),
            "--early_stopping_window_steps",
            str(args.early_stopping_window_steps),
        ]

        if args.task == "survival_clmbr":
            cmd += ["--clmbr_survival_dim", str(args.clmbr_survival_dim)]

        if args.start_from_checkpoint is not None:
            cmd += [
                "--start_from_checkpoint",
                args.start_from_checkpoint,
            ]

        subprocess.run(cmd)
