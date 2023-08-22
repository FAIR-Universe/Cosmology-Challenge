FROM registry.nersc.gov/library/nersc/mpi4py:3.1.3

RUN apt-get update && \
    apt-get install -y libgsl-dev

RUN pip install numpy scipy matplotlib notebook ipython ipykernel 
RUN pip install Cython==0.29.33
RUN pip install scikit-learn
RUN pip install mpsort nbodykit pmesh
RUN pip install fastpm

WORKDIR /usr/src/app
COPY src/ src/ 
COPY setup.py setup.py
RUN pip install .

RUN pip install emcee
WORKDIR /opt