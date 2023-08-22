#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J mpi4py_test
#SBATCH -o logs/mpi4py_test-%j.txt

srun -n 64 podman-hpc run --rm --mpi fpm:latest python3 scripts/tests/test_mpi.py
