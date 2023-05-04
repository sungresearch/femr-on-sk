#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --dependency=afterok:2707299
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 12 #number of cores to reserve, default is 1
#SBATCH --gpus=1
#SBATCH --mem=64000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/projects/lguo/femr-on-sk/scripts

## ablated after index
# python featurize.py \
#     "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/features/clmbr_stanford" \
#     --path_to_labels="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los" \
#     --clmbr \
#     --path_to_clmbr_data="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_models/clmbr_stanford" \
#     --force_use_extract="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/ablated_extract/femr_extract_ablated" \
#     --overwrite


## ablated before index
python featurize.py \
    "/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/features/clmbr_stanford_before_index" \
    --path_to_labels="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/labels/long_los" \
    --clmbr \
    --path_to_clmbr_data="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/clmbr_models/clmbr_stanford" \
    --force_use_extract="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/ablated_extract_before_index/femr_extract_ablated" \
    --overwrite

    
