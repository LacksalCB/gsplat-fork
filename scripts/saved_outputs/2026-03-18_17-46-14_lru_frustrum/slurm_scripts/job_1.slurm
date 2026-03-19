#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/2026-03-18_17-46-14/slurm_out/%x-%j.out
#SBATCH --error=outputs/2026-03-18_17-46-14/slurm_err/%x-%j.err
#SBATCH --constraint="a6000"

module purge
module load cuda
module load nsight-systems

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg

cd /scratch/rhm4nj/gpu_arch/gsplat-fork/examples
which nvcc

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
ENV_NAME="v100-env"
ENV_DIR="/scratch/rhm4nj/gpu_arch/envs/$ENV_NAME"

echo "GPU: $GPU_NAME"
echo "Env: $ENV_DIR"

echo "Reusing existing environment: $ENV_DIR"
source "$ENV_DIR/bin/activate"

echo 'which python (after):'
which python

echo 'working in dir:'
pwd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo 'Running nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --force-overwrite=true \
    --output=/scratch/rhm4nj/gpu_arch/gsplat-fork/scripts/slurm/outputs/2026-03-18_17-46-14/results/gsplat_profile_bicycle \
    bash /scratch/rhm4nj/gpu_arch/gsplat-fork/examples/benchmarks/custom_mod.sh \
    bicycle'

nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --force-overwrite=true \
    --output=/scratch/rhm4nj/gpu_arch/gsplat-fork/scripts/slurm/outputs/2026-03-18_17-46-14/results/gsplat_profile_bicycle \
    bash /scratch/rhm4nj/gpu_arch/gsplat-fork/examples/benchmarks/custom_mod.sh \
    bicycle