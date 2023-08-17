#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J egd_test
#SBATCH -o logs/egd_test-%j.txt

srun -n 1 podman-hpc run --rm --volume /pscratch:/pscratch fpm:latest python3 scripts/tests/test_egd.py
