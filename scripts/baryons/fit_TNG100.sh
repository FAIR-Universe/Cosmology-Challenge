#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -t 00:10:00
#SBATCH -J fit_baryons
#SBATCH -o logs/fit_baryons-%j.txt

srun -n 1 podman-hpc run --mpi --rm \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir \
    --volume $PWD/scripts:/scripts \
    --volume $PWD/data:/data \
    --volume $PWD/plots:/plots \
    fpm:latest python3 /scripts/baryons/fit_TNG100.py \

