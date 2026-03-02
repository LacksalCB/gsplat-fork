
module purge
module load cuda
module load nsight-systems

# Activate your environment
source ~/.bashrc
source /bigtemp/rhm4nj/gpu_arch/project/gsplat-env/bin/activate

which nvcc

cd /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/examples

# Optional: reduce profiler noise
export OMP_NUM_THREADS=1

# Profile with nsys
nsys profile \
    --trace=cuda,osrt,nvtx \
	--sample=cpu \
	--force-overwrite=true \
	--output=gsplat_profile_test \
	bash /bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/examples/benchmarks/basic.sh \
	--output results/profile_test

