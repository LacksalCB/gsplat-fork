#!/usr/bin/env bash

module load gcc 
module load python/3.12.3

source ~/gpu_6501/research/gsplat-env/bin/activate

echo "Entered VM"
pwd

pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu128


python -c "import torch; print(torch.version.cuda)"

pip install numpy ninja jaxtyping rich imageio

pip install --no-build-isolation -e ~/gpu_6501/research/gsplat

pip install -r ~/gpu_6501/research/gsplat/examples/requirements.txt --no-build-isolation

#python ~/gpu_6501/research/gsplat/examples/datasets/download_dataset.py
