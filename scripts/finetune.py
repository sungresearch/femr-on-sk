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
    parser.add_argument("--finetune_task_layer", default=False, action="store_true")
    parser.add_argument("--path_to_task_batches", type=str, default=None)
    parser.add_argument("--path_to_og_clmbr_data", type=str, required=True)
    parser.add_argument("--freeze_weights", default=False, action="store_true")
    parser.add_argument("--num_batch_threads", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_iter", type=int, default=None)

    args = parser.parse_args()

    PATH_PATIENT_DATABASE = path_extract
    PATH_OG_CLMBR_DATA = args.path_to_og_clmbr_data
    PATH_OG_DICTIONARY = os.path.join(PATH_OG_CLMBR_DATA, "dictionary")
    PATH_OG_MODEL = os.path.join(PATH_OG_CLMBR_DATA, "clmbr_model")

    FINETUNE_TASK_LAYER = args.finetune_task_layer
    FREEZE_WEIGHTS = args.freeze_weights
    PATH_TASK_BATCHES = args.path_to_task_batches

    PATH_OUTPUT_DIR = args.path_to_output_dir
    PATH_DICTIONARY = os.path.join(PATH_OUTPUT_DIR, "dictionary")
    PATH_BATCHES = os.path.join(PATH_OUTPUT_DIR, "clmbr_batches")
    PATH_MODEL = os.path.join(PATH_OUTPUT_DIR, "clmbr_model")

    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    # Copy dictionary
    if not os.path.exists(PATH_DICTIONARY):
        shutil.copyfile(PATH_OG_DICTIONARY, PATH_DICTIONARY)

    # load model config
    model_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))[
        "transformer"
    ]

    # Create batches
    if not FINETUNE_TASK_LAYER and PATH_TASK_BATCHES is not None:
        if args.overwrite and os.path.exists(PATH_BATCHES):
            shutil.rmtree(PATH_BATCHES)

        # We need to grab "transformer_vocab_size" from the original CLMBR
        # loader_config.msgpack. Alternatively, we can get this from model config.msgpack
        clmbr_config = read_msgpack(os.path.join(PATH_OG_MODEL, "config.msgpack"))
        vocab_size = clmbr_config["transformer"]["vocab_size"]

        if not os.path.exists(PATH_BATCHES):
            subprocess.run(
                [
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
                ]
            )

    # Fine-tune Full
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
            PATH_BATCHES if not FINETUNE_TASK_LAYER else PATH_TASK_BATCHES,
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
            model_config["n_heads"],
            "--n_layers",
            model_config["n_layers"],
            "--attention_width",
            model_config["attention_width"],
            "--num_batch_threads",
            str(args.num_batch_threads),
            "--hidden_size",
            model_config["hidden_size"],
            "--intermediate_size",
            model_config["intermediate_size"],
        ]

        if args.freeze_weights:
            cmd += ["--freeze_weights"]

        subprocess.run(cmd)
