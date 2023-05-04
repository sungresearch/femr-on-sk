"""
In this sanity check, we are using CLMBR features to predict arbitrary length 
of stay days (even / odd). Expects CLMBR features to perform 
similarly as count-based models, and all models should perform poorly.
"""
import argparse
import datetime
import os
import pickle
import time
import subprocess
import pdb

from typing import Optional, Callable, List
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from femr import Event, Patient
from femr.datasets import PatientDatabase
from femr.extension import datasets as extension_datasets
from femr.labelers.core import NLabelsPerPatientLabeler
from femr.labelers.omop_lab_values import HypoglycemiaLabValueLabeler
from femr.labelers.core import Label, Labeler, LabelType
from femr.labelers.omop import (
    WithinVisitLabeler,
    get_death_concepts,
    get_inpatient_admission_discharge_times,
    get_inpatient_admission_events,
    move_datetime_to_end_of_day,
    map_omop_concept_codes_to_femr_codes,
)
from femr.featurizers import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as auroc

from src.utils import (
    save_to_pkl, read_pkl, read_msgpack, save_to_msgpack, 
    load_features, hash_pids
)


class RandomGlucoseLabeler(HypoglycemiaLabValueLabeler):
    """assign severe based on whether glucose value is even"""

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        value = int(float(raw_value))
        if value % 2 == 0:
            return "severe"
        return "normal"
        

class RandLOSLabeler(Labeler):

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        prediction_time_adjustment_func: Callable = move_datetime_to_end_of_day,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.prediction_time_adjustment_func = prediction_time_adjustment_func

    def label(self, patient: Patient) -> List[Label]:
        """
        True if LOS is an even number
        """
        labels: List[Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            is_even_los: bool = (discharge_time - admission_time).days % 2 == 0
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)

            # exclude if discharge or death occurred before prediction time
            death_concepts: Set[str] = map_omop_concept_codes_to_femr_codes(self.ontology, get_death_concepts())
            death_times: List[datetime.datetime] = []
            for e in patient.events:
                if e.code in death_concepts:
                    death_times.append(e.start)

            if prediction_time > discharge_time:
                continue

            if death_times and prediction_time > min(death_times):
                continue

            labels.append(Label(prediction_time, is_even_los))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run femr labeler")
    parser.add_argument(
        "--path_to_patient_database",
        type=str,
        help="Path to femr PatientDatabase.",
        default="/hpf/projects/lsung/data/lguo/omop_extract_v6"
    )

    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        help=("Path to save labeled_patients.pkl"),
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/rand_los_prediction"
    )
    
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    # predefine paths
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    PATH_LABELS = os.path.join(args.path_to_output_dir, "labeled_patients.pkl")
    
    PATH_COUNT_FEATURES = os.path.join(args.path_to_output_dir, "features_count.pkl")
    PATH_SK_CLMBR_FEATURES = os.path.join(args.path_to_output_dir, "features_clmbr_sk.pkl")
    PATH_STANFORD_CLMBR_FEATURES = os.path.join(args.path_to_output_dir, "features_clmbr_stanford.pkl")
    
    PATH_COUNT_RESULTS = os.path.join(args.path_to_output_dir, "results_count.pkl")
    PATH_SK_CLMBR_RESULTS = os.path.join(args.path_to_output_dir, "results_clmbr_sk.pkl")
    PATH_STANFORD_CLMBR_RESULTS = os.path.join(args.path_to_output_dir, "results_clmbr_stanford.pkl")
    
    for path in [
        PATH_LABELS, PATH_COUNT_FEATURES, PATH_SK_CLMBR_FEATURES, 
        PATH_STANFORD_CLMBR_FEATURES, PATH_COUNT_RESULTS, 
        PATH_SK_CLMBR_RESULTS, PATH_STANFORD_CLMBR_RESULTS
    ]:
        if args.overwrite and os.path.exists(path):
            os.remove(path)
    
    
    # Load PatientDatabase + Ontology
    database = PatientDatabase(args.path_to_patient_database)
    ontology = database.get_ontology()
    labeler = RandLOSLabeler(ontology)
    labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=1)
                                   
    
    # label
    print("labeling")
    labeled_patients = labeler.apply(
        path_to_patient_database=args.path_to_patient_database,
        num_threads=4,
    )
    save_to_pkl(labeled_patients, os.path.join(PATH_LABELS))
        
    
    # featurize, train, eval
    
    # - count
    # -- featurize
    print("obtaining count features")
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizers = FeaturizerList([age, count])
    
    featurizers.preprocess_featurizers(
        args.path_to_patient_database,
        labeled_patients,
        4
    )
    
    featurized_patients = featurizers.featurize(
        args.path_to_patient_database, 
        labeled_patients, 
        4,
    )
    
    save_to_pkl(featurized_patients,os.path.join(PATH_COUNT_FEATURES))
    
    # -- train model
    X_train, y_train = load_features(
        args.path_to_patient_database,
        PATH_COUNT_FEATURES,
        PATH_LABELS,
        "count", 
        is_train=True
    )
    
    m = LogisticRegressionCV(
        Cs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 
        cv = 5,
        scoring = "neg_log_loss",
        max_iter = 10000,
        n_jobs = 4,
        refit = True
    )
    
    print("fitting count model")
    m.fit(X_train, y_train)
    
    # -- eval
    print("evaluating count model")
    X_test, y_test = load_features(
        args.path_to_patient_database,
        PATH_COUNT_FEATURES,
        PATH_LABELS,
        "count",
        is_eval=True,
    )
    
    preds = m.predict_proba(X_test)[:,1]
    results = {
        "labels": y_test,
        "predictions": preds,
        "model": m,
        "auroc": auroc(y_test, preds)
    }
    print("saving count results")
    save_to_pkl(results, PATH_COUNT_RESULTS)
    
    
    # - clmbr SK
    # -- featurize
    PATH_CLMBR_DATA = "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_sk/CLMBR_learning_rate_1e-4_max_iter_1000000_rotary_type_per_head"
    PATH_CLMBR_BATCHES = os.path.join(args.path_to_output_dir, "clmbr_sk_batches")
    model_config = read_msgpack(os.path.join(PATH_CLMBR_DATA, "clmbr_model", "config.msgpack"))
    vocab_size = model_config["transformer"]["vocab_size"]
    
    cmd = [
        "clmbr_create_batches",
        PATH_CLMBR_BATCHES, 
        "--data_path", args.path_to_patient_database,
        "--dictionary", os.path.join(PATH_CLMBR_DATA, "dictionary"),
        "--task", "labeled_patients",
        "--labeled_patients_path", PATH_LABELS,
        "--transformer_vocab_size", str(vocab_size)
    ]
    subprocess.run(cmd)
    
    subprocess.run(
        [
            "clmbr_compute_representations",
            PATH_SK_CLMBR_FEATURES, 
            "--data_path", args.path_to_patient_database,
            "--batches_path", PATH_CLMBR_BATCHES,
            "--model_dir", os.path.join(PATH_CLMBR_DATA, "clmbr_model"),
        ]
    )
    # -- train
    X_train, y_train = load_features(
        args.path_to_patient_database,
        PATH_SK_CLMBR_FEATURES,
        PATH_LABELS,
        "clmbr", 
        is_train=True
    )
    
    m = LogisticRegressionCV(
        Cs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 
        cv = 5,
        scoring = "neg_log_loss",
        max_iter = 10000,
        n_jobs = 4,
        refit = True
    )
    
    print("fitting CLMBR_SK adapter model")
    m.fit(X_train, y_train)
    
    # -- eval
    print("evaluating CLMBR_SK adapter model")
    X_test, y_test = load_features(
        args.path_to_patient_database,
        PATH_SK_CLMBR_FEATURES,
        PATH_LABELS,
        "clmbr",
        is_eval=True,
    )
    
    preds = m.predict_proba(X_test)[:,1]
    results = {
        "labels": y_test,
        "predictions": preds,
        "model": m,
        "auroc": auroc(y_test, preds)
    }
    print("saving SK clmbr results")
    save_to_pkl(results, PATH_SK_CLMBR_RESULTS)
    
    
    # - clmbr stanford
    # -- featurize
    PATH_CLMBR_DATA = "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_stanford"
    PATH_CLMBR_BATCHES = os.path.join(args.path_to_output_dir, "clmbr_stanford_batches")
    model_config = read_msgpack(os.path.join(PATH_CLMBR_DATA, "clmbr_model", "config.msgpack"))
    vocab_size = model_config["transformer"]["vocab_size"]
    
    cmd = [
        "clmbr_create_batches",
        PATH_CLMBR_BATCHES, 
        "--data_path", args.path_to_patient_database,
        "--dictionary", os.path.join(PATH_CLMBR_DATA, "dictionary"),
        "--task", "labeled_patients",
        "--labeled_patients_path", PATH_LABELS,
        "--transformer_vocab_size", str(vocab_size)
    ]
    subprocess.run(cmd)
    
    subprocess.run(
        [
            "clmbr_compute_representations",
            PATH_STANFORD_CLMBR_FEATURES, 
            "--data_path", args.path_to_patient_database,
            "--batches_path", PATH_CLMBR_BATCHES,
            "--model_dir", os.path.join(PATH_CLMBR_DATA, "clmbr_model"),
        ]
    )
    # -- train
    X_train, y_train = load_features(
        args.path_to_patient_database,
        PATH_STANFORD_CLMBR_FEATURES,
        PATH_LABELS,
        "clmbr", 
        is_train=True
    )
    
    m = LogisticRegressionCV(
        Cs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 
        cv = 5,
        scoring = "neg_log_loss",
        max_iter = 10000,
        n_jobs = 4,
        refit = True
    )
    
    print("fitting CLMBR_Stanford adapter model")
    m.fit(X_train, y_train)
    
    # -- eval
    print("evaluating CLMBR_Stanford adapter model")
    X_test, y_test = load_features(
        args.path_to_patient_database,
        PATH_STANFORD_CLMBR_FEATURES,
        PATH_LABELS,
        "clmbr",
        is_eval=True,
    )
    
    preds = m.predict_proba(X_test)[:,1]
    results = {
        "labels": y_test,
        "predictions": preds,
        "model": m,
        "auroc": auroc(y_test, preds)
    }
    print("saving STANFORD clmbr results")
    save_to_pkl(results, PATH_STANFORD_CLMBR_RESULTS)