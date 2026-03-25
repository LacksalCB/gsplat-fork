#!/bin/bash

BASE_DIR=~/gpu_6501/gsplat-fork
TIME=$(date +%y%m%d_%h%m%s)
INPUT=$1
OUTPUT=${TIME}/results/profile_gsplat-$1
RESULT_DIR="${BASE_DIR}/examples/results_slurm/gsplat_results_${TIME}"


module load cuda
module load nsight-systems

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg

cd $BASE_DIR/examples
which nvcc

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
ENV_NAME=".venv"
ENV_DIR="$BASE_DIR/$ENV_NAME"

echo "GPU: $GPU_NAME"
echo "Env: $ENV_DIR"

echo "Reusing existing environment: $ENV_DIR"
source "$ENV_DIR/bin/activate"

echo 'which python (after):'
which python

echo 'working in dir:'
pwd

export OMP_NUM_THREADS=4

time nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --force-overwrite=true \
    --output=$BASE_DIR/scripts/slurm/outputs/$OUTPUT \
    bash $BASE_DIR/examples/benchmarks/custom_mod.sh \
    	$1 \
    	$BASE_DIR/scripts/slurm/outputs/$OUTPUT \
		2 \
		--optimizer-stride 4 \
		--enable-frustum-culling \
		--enable-input-cache
    
