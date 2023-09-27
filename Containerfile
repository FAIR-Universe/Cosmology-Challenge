FROM registry.nersc.gov/library/nersc/mpi4py:3.1.3

RUN apt-get update &&               \
    apt-get install -y libgsl-dev   \
                        git

RUN pip install Cython==0.29.33
RUN pip install numpy           \
                scipy           \
                notebook        \
                ipython         \
                ipykernel       \ 
                scikit-learn    \
                mpsort          \
                nbodykit        \
                fastpm          \
                emcee           \
                corner          \
                h5py

RUN pip install --upgrade matplotlib

WORKDIR /usr/src/app
COPY src/ src/
COPY setup.py setup.py
RUN pip install .

WORKDIR /usr/src/pmesh
COPY pmesh/ pmesh/
WORKDIR /usr/src/pmesh/pmesh
RUN pip install .

WORKDIR /opt