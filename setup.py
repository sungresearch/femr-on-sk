from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="lawrence",
    author_email="lawrence.guo@sickkids.ca",
    packages=find_packages(),
    install_requires=[
        "pandas ~= 2.0",
        "matplotlib ~= 3.7",
        "matplotlib-venn ~= 0.11",
        "seaborn ~= 0.12",
        "dask[complete] ~= 2023.4",
        "dask-sql[complete] ~= 2023.6",
        "lightgbm ~= 3.3",
        "simple_slurm ~= 0.2",
    ],
)
