#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --dependency=afterok:2707316
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 32 #number of cores to reserve, default is 1
#SBATCH --mem=128000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/projects/lguo/femr-on-sk/scripts

# drop after index date
# python train_adapter.py \
#     "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/adapter_models/clmbr_stanford" \
#     --path_to_labels="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los" \
#     --path_to_features="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/features/clmbr_stanford" \
#     --feature_type="clmbr" \
#     --overwrite
    

# python evaluate_adapter.py \
#     "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/evaluate/clmbr_stanford" \
#     --path_to_model="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/adapter_models/clmbr_stanford" \
#     --overwrite


# drop before index date
python train_adapter.py \
    "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/adapter_models/clmbr_stanford_before_index" \
    --path_to_labels="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los" \
    --path_to_features="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/features/clmbr_stanford_before_index" \
    --feature_type="clmbr" \
    --overwrite
    

python evaluate_adapter.py \
    "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/evaluate/clmbr_stanford_before_index" \
    --path_to_model="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/adapter_models/clmbr_stanford_before_index" \
    --overwrite
    
