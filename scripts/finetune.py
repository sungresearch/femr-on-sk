import os
import pickle
import subprocess
import msgpack
import shutil
import argparse

import numpy as np

from src.utils import read_msgpack
from src.default_paths import path_extract

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pretrain CLMBR")
    parser.add_argument(
        "path_to_output_dir",
        type=str,
        help=("Path to save CLMBR data"
        ),
    )
    parser.add_argument("--overwrite", default=False, action='store_true')
    parser.add_argument("--finetune_last_layer", default=False, action='store_true')
    parser.add_argument("--path_to_og_clmbr_data", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--rotary_type", type=str, required=True)
    parser.add_argument("--num_batch_threads", type=int, default=3)
    parser.add_argument("--token_dropout", type=float, default=0)
    parser.add_argument("--internal_dropout", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=768, help="Transformer hidden size")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Transformer intermediate layer size")
    parser.add_argument("--n_heads", type=int, default=12, help="Transformer # of heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Transformer # of layers")
    parser.add_argument("--attention_width", type=int, default=512, help="Transformer attention width.")
    
    args = parser.parse_args()
    
    PATH_TO_PATIENT_DATABASE = path_extract
    PATH_TO_OG_CLMBR_DATA = args.path_to_og_clmbr_data
    PATH_OG_DICTIONARY = os.path.join(PATH_TO_OG_CLMBR_DATA, "dictionary")
    PATH_OG_MODEL = os.path.join(PATH_TO_OG_CLMBR_DATA, "clmbr_model")
    
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    PATH_DICTIONARY = os.path.join(PATH_TO_OUTPUT_DIR, "dictionary")
    PATH_BATCHES = os.path.join(PATH_TO_OUTPUT_DIR, "clmbr_batches")
    PATH_MODEL = os.path.join(PATH_TO_OUTPUT_DIR, "clmbr_model")
    
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    
    # Copy dictionary
    if not os.path.exists(PATH_DICTIONARY):
        shutil.copyfile(
            PATH_OG_DICTIONARY, 
            PATH_DICTIONARY
        )

    # Create batches
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
                "--data_path", PATH_TO_PATIENT_DATABASE,
                "--dictionary", PATH_DICTIONARY,
                "--transformer_vocab_size", str(vocab_size),
                "--task", "clmbr",
            ]
        )
        
    # Fine-tune Full
    if args.overwrite and os.path.exists(PATH_MODEL):
        shutil.rmtree(PATH_MODEL, ignore_errors=True)
        
    if not os.path.exists(PATH_MODEL):
        
        cmd = [
            "clmbr_train_model", 
            PATH_MODEL, 
            "--start_from_checkpoint", PATH_OG_MODEL,
            "--data_path", PATH_TO_PATIENT_DATABASE,
            "--batches_path", PATH_BATCHES,
            "--learning_rate", str(args.learning_rate),
            "--max_iter", str(args.max_iter),
            "--rotary_type", args.rotary_type,
            "--weight_decay", str(args.weight_decay),
            "--internal_dropout", str(args.internal_dropout),
            "--token_dropout", str(args.token_dropout),
            "--n_heads", str(args.n_heads),
            "--n_layers", str(args.n_layers),
            "--attention_width", str(args.attention_width),
            "--num_batch_threads", str(args.num_batch_threads),
            "--hidden_size", str(args.hidden_size),
            "--intermediate_size", str(args.intermediate_size),
        ]
        
        if args.finetune_last_layer:
            cmd += ["--finetune_last_layer"]
        
        subprocess.run(cmd)
        