from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="lawrence",
    author_email="lawrence.guo@sickkids.ca",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "matplotlib-venn",
        "seaborn",
        "dask[complete]",
        "dask-sql[complete]",
        "lightgbm",
        "simple_slurm",
    ],
)
