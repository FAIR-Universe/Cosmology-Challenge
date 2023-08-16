#!/bin/bash
module load python
module load gsl
# barebone
conda create -p /pscratch/sd/b/bthorne/conda/simulation python=3.7 numpy scipy matplotlib notebook ipython ipykernel Cython -y
conda config --append envs_dirs /pscratch/sd/b/bthorne/conda
conda info --envs
conda activate simulation
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
conda install scikit-learn -y
python -m ipykernel install --user --name simulation --display-name simulation
MPICC="cc" pip install --no-cache-dir --no-binary=mpsort mpsort
cd pmesh/
pip install -e .
cd ../
MPICC="cc" pip install --no-cache-dir --no-binary=nbodykit nbodykit