#!/bin/bash
#SBATCH --qos=regular
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -t 08:00:00
#SBATCH -J fit_baryons
#SBATCH -o logs/fit_baryons-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 128))
Nmesh=512
kmax=8
Nsamples=250
Nwalkers=4
scale_factor=0.4902

srun -n $total_processes podman-hpc run --mpi --rm                                          \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir                  \
    --volume $PWD/scripts:/scripts                                                          \
    --volume $PWD/data:/data                                                                \
    --volume $PWD/plots:/plots                                                              \
    fpm:latest python3 /scripts/baryons/fit_TNG100.py                                       \
    --kmax ${kmax}                                                                          \
    --Nmesh ${Nmesh}                                                                        \
    --Nwalkers ${Nwalkers}                                                                  \
    --Nsamples_per_walker ${Nsamples}                                                       \
    --scale_factor ${scale_factor}                                                          \
    --do_sampling

