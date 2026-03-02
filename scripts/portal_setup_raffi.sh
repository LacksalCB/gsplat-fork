#!/usr/bin/env bash

module load gcc 
module load python/3.12.3
module load cuda

source /bigtemp/rhm4nj/gpu_arch/project/gsplat-env/bin/activate

echo "Entered VM"
pwd
which python

pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu128


python -c "import torch; print(torch.version.cuda)"

pip install numpy ninja jaxtyping rich imageio

pip install --no-build-isolation -e /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork

pip install -r /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/examples/requirements.txt --no-build-isolation

#python ~/gpu_6501/research/gsplat/examples/datasets/download_dataset.py
