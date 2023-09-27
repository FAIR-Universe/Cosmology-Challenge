#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J plot_fit_baryons
#SBATCH -o logs/plot_fit_baryons-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 128))

srun -n $total_processes podman-hpc run --mpi --rm                                          \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir                  \
    --volume $PWD/scripts:/scripts                                                          \
    --volume $PWD/data:/data                                                                \
    --volume $PWD/plots:/plots                                                              \
    fpm:latest python3 /scripts/baryons/plot_samples.py