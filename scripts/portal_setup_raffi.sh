#!/usr/bin/env bash

WORKSPACE="/bigtemp/rhm4nj/gpu_arch/project"
GSPLAT_DIR="$WORKSPACE/gsplat-fork"

module load gcc
module load python/3.12.3
module load cuda
module load nsight-systems


source $WORKSPACE/gsplat-env/bin/activate
echo "Entered VM"
pwd
echo 'which python (setup):'
which python

pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu128


python -c "import torch; print(torch.version.cuda)"

pip install numpy ninja jaxtyping rich imageio

TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" pip install --no-build-isolation -e "$GSPLAT_DIR"

pip install -r "$GSPLAT_DIR/examples/requirements.txt" --no-build-isolation

#python ~/gpu_6501/research/gsplat/examples/datasets/download_dataset.py
