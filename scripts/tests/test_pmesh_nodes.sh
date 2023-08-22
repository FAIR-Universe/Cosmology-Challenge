#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -t 00:10:00
#SBATCH -J test_pmesh_nodes
#SBATCH -o logs/test_pmesh_nodes-%j.txt

export OMP_NUM_THREADS=1
total_processes=$((SLURM_JOB_NUM_NODES * 128))

echo "Running on $SLURM_NNODES nodes with $SLURM_CPUS_ON_NODE cores per node"
echo "Launching a total of $total_processes processes"

srun -n $total_processes podman-hpc run --rm --mpi \
    --volume $PWD/data:/data \
    --volume $PWD/plots:/plots \
    --volume $PWD/scripts:/scripts \
    --volume /pscratch/sd/b/bthorne/fairuniverse/hsc_dataset:/snapshot_dir \
    fpm:latest python3 /scripts/tests/test_pmesh_nodes.py \
    --output /plots/tests/test_pmesh_nodes_N${SLURM_NNODES}_J${SLURM_JOB_ID} \
    --title $SLURM_NNODES