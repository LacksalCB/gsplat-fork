#!/usr/bin/env bash

# source /bigtemp/rhm4nj/gpu_arch/project/gsplat-env/bin/activate
source /scratch/rhm4nj/gpu_arch/envs/a6000-env/bin/activate

module load gcc 
module load python/3.12.3
module load cuda

echo "Entered VM"
pwd
echo 'which python (setup):'
which python
which nvcc

pip install wheel
pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu128

echo 'which torch version:'
python -c "import torch; print(torch.version.cuda)"

pip install numpy ninja jaxtyping rich imageio

# pip install --no-build-isolation -e /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork
pip install --no-build-isolation -e /scratch/rhm4nj/gpu_arch/gsplat-fork

# pip install -r /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/examples/requirements.txt --no-build-isolation
pip install -r /scratch/rhm4nj/gpu_arch/gsplat-fork/examples/requirements.txt --no-build-isolation

#python ~/gpu_6501/research/gsplat/examples/datasets/download_dataset.py
