"""
In this sanity check, we are ablating patient timelines. Expectation
ablation of timeline after index date to not impact model performance,
whereas ablation of timeline before index date does impact model performance.
"""
import argparse
import datetime
import functools
import json
import logging
import os
import resource
import random
import shutil
from typing import Callable, Dict, Optional, Sequence, List
from femr.datasets import EventCollection, PatientCollection, RawEvent, RawPatient
from femr.labelers.core import Label

from src.utils import read_pkl


def ablate_timeline_after_index(
    patient: RawPatient,
    cohort: Dict[int, List[Label]],
) -> RawPatient:
    """
    Remove all events past prediction date Label.time.date()
    Only works if patient has a single label - by default this is the case
    """
    
    if patient.patient_id not in cohort:
        return patient
    
    index_date = cohort[patient.patient_id][0].time.date()
    new_events = []
    
    for event in patient.events:
        if event.start.date() <= index_date:
            new_events.append(event)
    
    patient.events = new_events
    patient.resort()
    
    return patient


def ablate_timeline_before_index(
    patient: RawPatient,
    cohort: Dict[int, List[Label]],
) -> RawPatient:
    """
    Remove randomly 50% of events before prediction date Label.time.date()
    Only works if patient has a single label - by default this is the case
    """
    
    if patient.patient_id not in cohort:
        return patient
    
    index_date = cohort[patient.patient_id][0].time.date()
    events_before = []
    events_after = []
    
    for i, event in enumerate(patient.events):
        # ignore first event (birthdate) and index event
        if i > 0 and event.start.date() < index_date:
            events_before.append(event)
        else:
            events_after.append(event)
            
    n_events_to_drop = int(0.5*len(events_before))
    random.shuffle(events_before)
    
    patient.events = events_before[n_events_to_drop:] + events_after
    patient.resort()
    
    return patient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timeline ablation experiment")

    parser.add_argument(
        "--path_to_patient_collection",
        type=str,
        help="Path to patient collection",
        default="/hpf/projects/lsung/data/femr_extract_temp/patients_cleaned"
    )
    
    parser.add_argument(
        "--target_location",
        type=str,
        help="Path to save ablated patient timelines",
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/ablated_extract"
    )
    
    parser.add_argument(
        "--cohort_path",
        type=str,
        help="Path to cohort - defaults to long LOS cohort",
        default="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los/labeled_patients.pkl"
    )
    
    parser.add_argument(
        "--omop_source",
        type=str,
        help="Path to omop source",
        default="/hpf/projects/lsung/data/omop_20230301_validated_csv",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=4,
    )
    
    parser.add_argument(
        "--drop_before_index",
        default=False,
        action="store_true",
        help="whether to drop events before index instead of after index",
    )
    
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
    )
    
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.target_location):
        shutil.rmtree(args.target_location, ignore_errors=True)
    
    if not os.path.exists(args.target_location):
        os.mkdir(args.target_location)
    
    cohort = read_pkl(args.cohort_path).get_patients_to_labels()
    cohort = {k:v for k,v in cohort.items() if v}

    patient_collection = PatientCollection(args.path_to_patient_collection)
    
    stats_dict = {}
    patient_collection = patient_collection.transform(
        os.path.join(args.target_location,"patients_ablated"),
        [
            functools.partial(
                ablate_timeline_before_index if args.drop_before_index else ablate_timeline_after_index, 
                cohort = cohort
            )
        ],
        num_threads=args.num_threads,
        stats_dict=stats_dict
    )
        
    patient_collection.to_patient_database(
        os.path.join(args.target_location,"femr_extract_ablated"),
        args.omop_source,
        num_threads=args.num_threads,
        delimiter="\t",
    ).close()