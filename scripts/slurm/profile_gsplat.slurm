#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load modules (adjust to UVA cluster modules)
module purge
module load cuda
module load nsight-systems

# Activate your environment
source ~/.bashrc
conda activate gsplat-env

# Optional: reduce profiler noise
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Profile with nsys
nsys profile \
    --trace=cuda,osrt,nvtx \
	--sample=cpu \
	--force-overwrite=true \
	--output=gsplat_profile_${SLURM_JOB_ID} \
	bash ~/gpu_6501/research/gsplat/examples/benchmark/basic.sh \
	--output results/profile_test

