#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J egd_test_spectra
#SBATCH -o logs/egd_spectra_test-%j.txt

srun -n 512 podman-hpc run --mpi --rm                       \
    --volume /pscratch:/pscratch                            \
    --volume $PWD/scripts:/scripts                          \
    --volume $PWD/plots:/plots                              \
    fpm:latest python3 /scripts/tests/test_egd_spectra.py
