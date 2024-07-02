# FEMR on SK OMOP

Running [Foundation models for Electronic Medical Records (FEMR)](https://github.com/som-shahlab/femr) on SickKids (SK) OMOP CDM.

This codebase contains scripts to:
1. compute count-based features and train logistic regression models
2. pretrain [CLMBR](https://www.sciencedirect.com/science/article/pii/S1532046420302653) foundation model on SK data
3. conduct fine-tuning and linear-probing using CLMBR features
4. adaptation of the publicly available [CLMBR-T-base](https://huggingface.co/StanfordShahLab/clmbr-t-base) pretrained on 2.57 million deidentified EHRs from Stanford Medicine.

#### Evaluation settings:
- Benchmark GBM and CLMBR models
- Few-shots settings
- Pretraining from scratch and continued pretraining on subsamples

## Installation

#### Create conda environment
```
conda create -p /path/to/env python=3.10 jupyterlab -c conda-forge -y
```

#### Clone repository
```
git clone https://github.com/sungresearch/femr-on-sk.git
```

#### Install package
```
cd femr-on-sk
pip install -e .
```

#### Install FEMR
```
pip install femr==0.1.16
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "femr_cuda[models]==0.1.16"
```

#### Post installation

Check `femr-on-sk/src/default_paths.py` for correct default root paths to the project directory and the FEMR patient database (extract).

## Precommit checks

If you wish to `git commit` to the repository, please run the following commands to ensure that your code is formatted correctly.

#### Installation
```
conda install pre-commit -y
pre-commit install
```

#### Formatting Checks
```
pre-commit run --all-files
```
