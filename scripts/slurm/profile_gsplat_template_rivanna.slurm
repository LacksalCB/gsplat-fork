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

cd {project_dir}/examples
which nvcc

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
ENV_NAME="{gpu_type}-env"
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

time nsys profile \
    --trace={trace} \
    --sample={sample} \
    --force-overwrite={force_overwrite} \
    --output={profile_output} \
    bash {script_path} \
    {scenes} \
    {profile_output} \
    {gpu_num} \
    {args}