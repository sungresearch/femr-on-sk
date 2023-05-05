#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo@sickkids.ca
#SBATCH --time=7-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH -n 12 #number of cores to reserve, default is 1
#SBATCH --mem=64000 # in MegaBytes. default is 8 GB
#SBATCH --gpus=1
#SBATCH --constraint="AlmaLinux8"
#SBATCH --error=/hpf/projects/lsung/phi/projects/lguo/femr-on-sk/logs/error-sbatchjob.%J.err
#SBATCH --output=/hpf/projects/lsung/phi/projects/lguo/femr-on-sk/logs/out-sbatchjob.%J.out

source activate /hpf/projects/lsung/envs/lguo/femr
cd /hpf/projects/lsung/phi/projects/lguo/femr-on-sk/scripts

## pretrain CLMBR_SK
#python run.py --pretrain="pretrain/sk.yml"

## featurize
#python run.py --featurize="featurize/clmbr_stanford.yml"
#python run.py --featurize="featurize/clmbr_sk.yml"

## continue_pretrain CLMBR_STANFORD
#python run.py --continue_pretrain="continue_pretrain/stanford.yml"
#python run.py --featurize="featurize/clmbr_stanford_cp.yml"

## finetune CLMBR_STANFORD
#python run.py --finetune="finetune/clmbr_stanford.yml"

## two-step finetune CLMBR_STANFORD
#python run.py --finetune="finetune/clmbr_stanford_two_step.yml"
