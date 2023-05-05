# FEMR on SK OMOP

Running [Foundation models for Electronic Medical Records (FEMR)](https://github.com/som-shahlab/femr) on SickKids (SK) OMOP CDM.

This codebase contains scripts to:
1. compute count-based features and train logistic regression models
2. pretrain [CLMBR](https://www.sciencedirect.com/science/article/pii/S1532046420302653) foundation model on SK data
3. conduct fine-tuning and linear-probing using CLMBR features
4. adaption of CLMBR pretrained on de-identified [Stanford Electronic Health Records in OMOP (STARR OMOP)](https://med.stanford.edu/starr-omop.html)

#### Evaluation settings:
- Comparison across models and adaptation strategies
- Few-shot training

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
pip install femr
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "femr_cuda[models]"
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

## Note for running SK FEMR ETL

The ETL from SK-OMOP to FEMR Patient Database will break (as of 2023-05-05) if you follow the FEMR installation guide above. You'd need to modify the `femr/src/femr/extractors/omop.py` line 211 to change `concept_id_field="piton_visit_detail_concept_id"` to `concept_id_field="visit_detail_concept_id"`. If you are running the code on HPC, the path to an existing SK FEMR patient database is specified in the default paths.
