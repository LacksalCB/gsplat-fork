#!/bin/bash

BASE_DIR=~/gpu_6501/gsplat-fork
TIME=$(date +%y%m%d_%h%m%s)
OUTPUT=${TIME}/results/profile_gsplat-rubble
RESULT_DIR="${BASE_DIR}/examples/results_slurm/gsplat_results_${TIME}"


#DEFAULTS=(bicycle bonsai counter garden kitchen room stump) 
#if [ "$#" -eq 0 ]; then
#    INPUTS=("${DEFAULTS[@]}")
#else
#	INPUTS=("$@")
#fi

module load cuda
module load nsight-systems

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg
export CUDA_VISIBLE_DEVICES=0 

cd $BASE_DIR
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

echo "DATA_DIR: '/bigtemp/sgr4vf/rubble-colmap'"
python -c "import os; print(os.path.exists('/bigtemp/sgr4vf/rubble-colmap/sparse/0'))"

time nsys profile \
    	--trace=cuda,osrt,nvtx \
		--cuda-memory-usage=true \
    	--sample=cpu \
    	--force-overwrite=true \
    	--output=$BASE_DIR/scripts/slurm/outputs/$OUTPUT \
    	python -u examples/trainer.py default \
		--disable_viewer \
		--data_factor 1 \
        --render_traj_path ellipse \
        --data_dir /bigtemp/sgr4vf/rubble-colmap \
        --result_dir $BASE_DIR/scripts/slurm/outputs/$OUTPUT \
        --eval_steps 0 \
		--optimizer-stride 4 \
		--enable-frustum-culling \
		--enable-input-cache
#
#for INPUT in "${INPUTS[@]}"; do
#	time nsys profile \
#    	--trace=cuda,osrt,nvtx \
#		--cuda-memory-usage=true \
    #	--sample=cpu \
   # 	--force-overwrite=true \
   # 	--output=$BASE_DIR/scripts/slurm/outputs/$OUTPUT \
   # 	bash $BASE_DIR/examples/benchmarks/custom_mod.sh \
   # 		$INPUT \
   # 		$BASE_DIR/scripts/slurm/outputs/$OUTPUT \
#			2 \
#			--optimizer-stride 4 \
#			--enable-frustum-culling \
#			--enable-input-cache
 
#done   
