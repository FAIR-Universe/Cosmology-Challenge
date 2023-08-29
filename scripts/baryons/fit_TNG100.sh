#!/bin/bash
#SBATCH --qos=regular
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -t 04:00:00
#SBATCH -J fit_baryons
#SBATCH -o logs/fit_baryons-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 128))

srun -n $total_processes podman-hpc run --mpi --rm                          \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir  \
    --volume $PWD/scripts:/scripts                                          \
    --volume $PWD/data:/data                                                \
    --volume $PWD/plots:/plots                                              \
    fpm:latest python3 /scripts/baryons/fit_TNG100.py                       \
    --output_stub kmax10_Nmesh1024_${SLURM_JOB_ID}                          \ 
    --kmax 10                                                               \                  
    --Nmesh 1024

