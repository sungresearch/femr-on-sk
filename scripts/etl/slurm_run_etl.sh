#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH -n 24 #number of cores to reserve, default is 1
#SBATCH --mem=128000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/sk-femr/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/sk-femr/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr

PATH_SOURCE="/hpf/projects/lsung/data/omop_20230301_validated_csv"
PATH_DESTINATION="/hpf/projects/lsung/data/femr_extract"
PATH_TEMP="/hpf/projects/lsung/data/femr_extract_temp"

etl_sickkids_omop \
    $PATH_SOURCE \
    $PATH_DESTINATION \
    $PATH_TEMP \
    --num_threads=12
