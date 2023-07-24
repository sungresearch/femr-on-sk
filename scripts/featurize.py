import argparse
import datetime
import os
import shutil
import time
import subprocess

from femr.featurizers import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers import load_labeled_patients
from src.io import save_to_pkl, read_msgpack
from src.default_paths import path_extract


def run_count_featurizers(args):
    """
    Featurize using count-based featurizers.
    Default settings for count featurizer:
        is_ontology_expansion=False
        time_bins=[
            datetime.timedelta(days=1),
            datetime.timedelta(days=7),
            datetime.timedelta(days=36500),
        ]
        numeric_value_decile=True
    """
    if args.force_use_extract is None:
        PATH_TO_PATIENT_DATABASE: str = path_extract
    else:
        PATH_TO_PATIENT_DATABASE: str = args.force_use_extract

    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_LABELS: str = args.path_to_labels
    NUM_THREADS: int = args.num_threads

    print(
        f"\n\
    PatientDatabase path: {PATH_TO_PATIENT_DATABASE}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Labels path: {PATH_TO_LABELS}\n\
    Number of threads: {NUM_THREADS}\n\
    Use ontology extension: {args.is_ontology_expansion}\n\
    "
    )

    START_TIME = time.time()

    if args.overwrite and os.path.exists(PATH_TO_OUTPUT_DIR):
        shutil.rmtree(PATH_TO_OUTPUT_DIR, ignore_errors=True)

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    age = AgeFeaturizer()
    count = CountFeaturizer(
        is_ontology_expansion=args.is_ontology_expansion,
        time_bins=[
            datetime.timedelta(days=1),
            datetime.timedelta(days=7),
            datetime.timedelta(days=36500),
        ],
        numeric_value_decile=True,
    )
    featurizers = FeaturizerList([age, count])

    # Preprocess featurizer
    print("Preprocessing featurizer")
    labeled_patients = load_labeled_patients(
        os.path.join(PATH_TO_LABELS, "labeled_patients.csv")
    )

    featurizers.preprocess_featurizers(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )

    save_to_pkl(
        featurizers, os.path.join(PATH_TO_OUTPUT_DIR, "preprocessed_featurizers.pkl")
    )

    # Run featurizer
    print("Running featurizer")
    featurized_patients = featurizers.featurize(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )

    save_to_pkl(
        featurized_patients, os.path.join(PATH_TO_OUTPUT_DIR, "featurized_patients.pkl")
    )

    time_elapsed = int(time.time() - START_TIME)
    print(f"Finished in {time_elapsed} seconds")


def run_clmbr_featurizer(args):
    """
    Featurize using CLMBR
    """

    if args.force_use_extract is None:
        PATH_TO_PATIENT_DATABASE: str = path_extract
    else:
        PATH_TO_PATIENT_DATABASE: str = args.force_use_extract

    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_LABELS: str = args.path_to_labels

    PATH_TASK_BATCHES = os.path.join(PATH_TO_OUTPUT_DIR, "task_batches")
    PATH_FEATURES = os.path.join(PATH_TO_OUTPUT_DIR, "featurized_patients.pkl")
    PATH_DICTIONARY = os.path.join(args.path_to_clmbr_data, "dictionary")
    PATH_MODEL = os.path.join(args.path_to_clmbr_data, "clmbr_model")

    print(
        f"\n\
    PatientDatabase path: {PATH_TO_PATIENT_DATABASE}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Labels path: {PATH_TO_LABELS}\n\
    CLMBR data path: {args.path_to_clmbr_data}\n\
    "
    )

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # load model config to get vocab size
    model_config = read_msgpack(os.path.join(PATH_MODEL, "config.msgpack"))
    vocab_size = model_config["transformer"]["vocab_size"]

    # create task batches
    if args.overwrite and os.path.exists(PATH_TASK_BATCHES):
        shutil.rmtree(PATH_TASK_BATCHES, ignore_errors=True)

    if not os.path.exists(PATH_TASK_BATCHES):
        cmd = [
            "clmbr_create_batches",
            PATH_TASK_BATCHES,
            "--data_path",
            PATH_TO_PATIENT_DATABASE,
            "--dictionary",
            PATH_DICTIONARY,
            "--task",
            "labeled_patients",
            "--labeled_patients_path",
            os.path.join(PATH_TO_LABELS, "labeled_patients.csv"),
            "--transformer_vocab_size",
            str(vocab_size),
            "--val_start",
            str(70),
        ]

        subprocess.run(cmd)

    # compute representations
    if args.overwrite and os.path.exists(PATH_FEATURES):
        os.remove(PATH_FEATURES)

    if not os.path.exists(PATH_FEATURES):
        subprocess.run(
            [
                "clmbr_compute_representations",
                PATH_FEATURES,
                "--data_path",
                PATH_TO_PATIENT_DATABASE,
                "--batches_path",
                PATH_TASK_BATCHES,
                "--model_dir",
                PATH_MODEL,
            ]
        )


def run_motor_featurizer(args):
    """
    Featurize using MOTOR
    """

    if args.force_use_extract is None:
        PATH_TO_PATIENT_DATABASE: str = path_extract
    else:
        PATH_TO_PATIENT_DATABASE: str = args.force_use_extract

    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_LABELS: str = args.path_to_labels

    PATH_TASK_BATCHES = os.path.join(PATH_TO_OUTPUT_DIR, "task_batches")
    PATH_FEATURES = os.path.join(PATH_TO_OUTPUT_DIR, "featurized_patients.pkl")
    PATH_DICTIONARY = os.path.join(args.path_to_clmbr_data, "dictionary")
    PATH_SURVIVAL_DICTIONARY = os.path.join(
        args.path_to_clmbr_data, "survival_dictionary"
    )
    PATH_MODEL = os.path.join(args.path_to_clmbr_data, "clmbr_model")

    print(
        f"\n\
    PatientDatabase path: {PATH_TO_PATIENT_DATABASE}\n\
    Output path: {PATH_TO_OUTPUT_DIR}\n\
    Labels path: {PATH_TO_LABELS}\n\
    MOTOR data path: {args.path_to_clmbr_data}\n\
    "
    )

    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # load model config to get vocab size
    model_config = read_msgpack(os.path.join(PATH_MODEL, "config.msgpack"))
    vocab_size = model_config["transformer"]["vocab_size"]

    # create task batches
    if args.overwrite and os.path.exists(PATH_TASK_BATCHES):
        shutil.rmtree(PATH_TASK_BATCHES, ignore_errors=True)

    if not os.path.exists(PATH_TASK_BATCHES):
        cmd = [
            "clmbr_create_batches",
            PATH_TASK_BATCHES,
            "--data_path",
            PATH_TO_PATIENT_DATABASE,
            "--dictionary",
            PATH_DICTIONARY,
            "--clmbr_survival_dictionary_path",
            PATH_SURVIVAL_DICTIONARY,
            "--task",
            "labeled_patients",
            "--labeled_patients_path",
            os.path.join(PATH_TO_LABELS, "labeled_patients.csv"),
            "--transformer_vocab_size",
            str(vocab_size),
            "--val_start",
            str(70),
        ]

        if model_config["transformer"]["is_hierarchical"]:
            cmd += ["--is_hierarchical"]

        subprocess.run(cmd)

    # compute representations
    if args.overwrite and os.path.exists(PATH_FEATURES):
        os.remove(PATH_FEATURES)

    if not os.path.exists(PATH_FEATURES):
        subprocess.run(
            [
                "clmbr_compute_representations",
                PATH_FEATURES,
                "--data_path",
                PATH_TO_PATIENT_DATABASE,
                "--batches_path",
                PATH_TASK_BATCHES,
                "--model_dir",
                PATH_MODEL,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run featurizer")
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--path_to_labels", type=str)
    parser.add_argument("--overwrite", default=False, action="store_true")

    # arguments for count-based featurizer
    parser.add_argument("--count", default=False, action="store_true")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--is_ontology_expansion", action="store_true")

    # arguments for CLMBR featurizer
    parser.add_argument("--clmbr", default=False, action="store_true")
    parser.add_argument("--motor", default=False, action="store_true")
    parser.add_argument("--path_to_clmbr_data", type=str, default=None)

    parser.add_argument(
        "--force_use_extract",
        type=str,
        default=None,
        help="For sanity check. Use another extract than specified in default_paths",
    )

    args = parser.parse_args()

    if args.count:
        run_count_featurizers(args)

    if args.clmbr:
        run_clmbr_featurizer(args)

    if args.motor:
        run_motor_featurizer(args)
