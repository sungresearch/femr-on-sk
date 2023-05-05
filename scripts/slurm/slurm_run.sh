#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=3-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 24 #number of cores to reserve, default is 1
#SBATCH --mem=32000 # in MegaBytes. default is 8 GB
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/phi/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/phi/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/phi/projects/lguo/femr-on-sk/scripts

## run labelers
#python run.py --label="label.yml"

## run count featurizer
#python run.py --featurize="featurize/count.yml"
python run.py --train_adapter="train_adapter"
python run.py --train_adapter_few_shots="train_adapter_few_shots"
python run.py --evaluate="evaluate/adapter_models.yml"
python run.py --evaluate="evaluate/adapter_models_few_shots.yml"
