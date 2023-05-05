# Setup

#### Installation

Create conda environment 
```
conda create -p /path/to/env python=3.10 jupyterlab -c conda-forge -y
```

Clone repository
```
git clone https://github.com/sungresearch/femr-on-sk.git
```

Install package
```
cd femr-on-sk
pip install -e .
```

Install FEMR
```
pip install femr
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "femr_cuda[models]"
```

Once installed, check `femr-on-sk/src/default_paths.py` for correct default path to the `femr-on-sk` directory and the FEMR patient database. 

#### Note for running SK FEMR ETL

The ETL from SK-OMOP to FEMR Patient Database will break (as of 2023-05-05) if you follow the FEMR installation guide above. You'd need to modify the `femr/src/femr/extractors/omop.py` line 211 to change `concept_id_field="piton_visit_detail_concept_id"` to `concept_id_field="visit_detail_concept_id"`. If you are running the code on HPC, the path to an existing SK FEMR patient database is specified in the default paths. 