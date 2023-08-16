#!/bin/bash
#SBATCH -q regular
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH --job-name=TrainDemo
#SBATCH --nodes=1
#SBATCH --licenses=scratch
#SBATCH --gpus=1
module load pytorch/2.0.1
srun python scripts/baseline_model_train.py
