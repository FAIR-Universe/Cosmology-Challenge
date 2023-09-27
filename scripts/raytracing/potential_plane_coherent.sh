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
    --volume $PWD/data/raytracing:/data                                     \
    fpm:latest python3 /scripts/raytracing/potentialplane_coherent.py       \
    --simulation 0                                                          \
    --base_dir /snapshot_dir                                                \
    --data_dir /data                                                        \
    --snapshot_start 0                                                      \
    --snapshot_end 43                                                       \
    --cosmology_file /data/cosmology.txt                                    \
    --redshifts_file /data/redshifts.txt


