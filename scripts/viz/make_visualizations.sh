#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -t 00:10:00
#SBATCH -J viz
#SBATCH -o logs/viz-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 1))

echo "Running on $SLURM_NNODES nodes with $SLURM_CPUS_ON_NODE cores per node"
echo "Launching a total of $total_processes processes"

snapshot_plot_path=/plots/snapshot_z_evolution_nodes${SLURM_NNODES}_${SLURM_JOB_ID}.png

srun -n 128 podman-hpc run --rm --mpi \
    --volume $PWD/data:/data \
    --volume $PWD/plots:/plots \
    --volume $PWD/scripts:/scripts \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir \
    fpm:latest python3 /scripts/viz/egd_examples.py \
    --output_stub egd_examples \
    --kmax 8 \
    --Nmesh 512

# srun -n $total_processes podman-hpc run --rm --mpi \
#     --volume $PWD/data:/data \
#     --volume $PWD/plots:/plots \
#     --volume $PWD/scripts:/scripts \
#     --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir \
#     fpm:latest python3 /scripts/viz/snapshot_z_evolution.py \
#     --output $snapshot_plot_path

# srun podman-hpc run --rm \
#     --volume $PWD/data:/data \
#     --volume $PWD/plots:/plots \
#     --volume $PWD/scripts:/scripts \
#     fpm:latest python3 /scripts/viz/TNG100_Pkratio_z_evolution.py

