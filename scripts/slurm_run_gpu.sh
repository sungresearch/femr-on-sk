#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=5-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 12 #number of cores to reserve, default is 1
#SBATCH --mem=64000 # in MegaBytes. default is 8 GB
#SBATCH --gpus=1
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/projects/lguo/femr-on-sk/scripts

## pretrain CLMBR_SK
python run.py --pretrain="pretrain/sk.yml"

## featurize
python run.py --featurize="featurize/clmbr_stanford_ft_full.yml"

## finetune CLMBR_STANFORD
#python run.py --finetune="finetune/stanford_full.yml"
#python run.py --featurize="featurize/clmbr_stanford_ft_full.yml"


## sanity_check for rand glucose predictions
#python sanity_checks/rand_glucose_prediction.py --overwrite
