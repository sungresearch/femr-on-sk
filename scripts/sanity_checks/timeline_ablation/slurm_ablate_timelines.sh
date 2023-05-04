#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 32 #number of cores to reserve, default is 1
#SBATCH --mem=128000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/projects/lguo/femr-on-sk/scripts/sanity_checks/timeline_ablation

# drop events after index
# python create_ablated_timelines.py --overwrite --num_threads=32

# drop events before index
python create_ablated_timelines.py \
    --overwrite --num_threads=32 \
    --drop_before_index \
    --target_location="/hpf/projects/lsung/projects/lguo/femr-on-sk/data/sanity_checks/timeline_ablation/ablated_extract_before_index"

    
