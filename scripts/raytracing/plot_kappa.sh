#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J plot_kappa
#SBATCH -o logs/plot_kappa-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 128))


srun -n $total_processes podman-hpc run --mpi --rm                          \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir  \
    --volume $PWD/plots:/plots                                              \
    --volume $PWD/scripts:/scripts                                          \
    --volume $PWD/data/raytracing:/data                                     \
    fpm:latest python3 /scripts/raytracing/plot_kappa.py


