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

source ~/.bashrc
{load_env_cmd}
cd {project_dir}
which nvcc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nsys profile \
    --trace={trace} \
    --sample={sample} \
    --force-overwrite={force_overwrite} \
    --output={profile_output} \
    bash {script_path} \
    {script_args}