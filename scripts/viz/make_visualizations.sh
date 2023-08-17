#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J viz
#SBATCH -o logs/viz-%j.txt

srun podman-hpc run --rm \
    --volume $PWD/data:/data \
    --volume $PWD/plots:/plots \
    fpm:latest python3 scripts/viz/TNG100_Pkratio_z_evolution.py