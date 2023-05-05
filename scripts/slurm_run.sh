#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=1-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=job
#SBATCH --dependency=afterok:2728578
#SBATCH --nodes=1
#SBATCH -n 24 #number of cores to reserve, default is 1
#SBATCH --mem=32000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/projects/lguo/femr-on-sk/scripts

## run labelers
#python run.py --label="label.yml"

## run count featurizer
python run.py --featurize="featurize/count.yml"
python run.py --train_adapter="train_adapter/count_sk.yml"
python run.py --train_adapter_few_shots="train_adapter_few_shots"
python run.py --evaluate_adapter="evaluate_adapter"

# train adapter models
# python run.py --featurize="featurize/count.yml"
# python run.py \
#     --featurize="featurize/count_no_expansion.yml"\
#     --train_adapter="train_adapter/count_sk_no_expansion.yml" \
#     --train_adapter_reduced_samples="train_adapter_reduced_samples/count_sk_no_expansion.yml" \
#     --evaluate_adapter="evaluate_adapter"
