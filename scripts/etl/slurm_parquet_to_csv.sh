#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH -n 12 #number of cores to reserve, default is 1
#SBATCH --mem=32000 # in MegaBytes. default is 8 GB
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr

PATH_SOURCE="/hpf/projects/lsung/data/omop_20230301_validated"
PATH_DESTINATION="/hpf/projects/lsung/data/omop_20230301_validated_csv"
REPARTITION_SIZE="100MB"
#COMPRESSION="gzip"

python parquet_to_csv.py \
    --path_omop_parquet_source=$PATH_SOURCE \
    --path_omop_csv_destination=$PATH_DESTINATION \
    --overwrite \
    --repartition_size=$REPARTITION_SIZE
    #--compression=$COMPRESSION
