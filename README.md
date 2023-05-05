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