#!/bin/bash

#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --constraint="{gpu_type}"

module purge
module load cuda
module load nsight-systems

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg

# {load_env_cmd}
cd {project_dir}/examples
which nvcc

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
ENV_NAME="gsplat-env-$GPU_NAME"
ENV_DIR="/bigtemp/rhm4nj/gpu_arch/project/envs/$ENV_NAME"

echo "GPU: $GPU_NAME"
echo "Env: $ENV_DIR"

# Use the env specifically for the GPU type, to avoid conflicts with different CUDA versions
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating new environment: $ENV_DIR"
    python -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"

    echo 'which python (before install):'
    which python

    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; print(torch.version.cuda)"
    pip install numpy ninja jaxtyping rich imageio
    pip install --no-build-isolation /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork
    pip install -r /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/examples/requirements.txt --no-build-isolation
else
    echo "Reusing existing environment: $ENV_DIR"
    source "$ENV_DIR/bin/activate"
fi

pip install imageio
echo 'which python (after):'
which python

echo 'working in dir:'
pwd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nsys profile \
    --trace={trace} \
    --sample={sample} \
    --force-overwrite={force_overwrite} \
    --output={profile_output} \
    bash {script_path} \
    {script_args}