#!/bin/bash
# Usage: bash nsys_profile.sh [REPORT NAME]

PREFIX="~/nsys_reports/nsys_"
REPORT=${PREFIX}${1}

echo "Creating $REPORT"

#Toggle if X gpu is busy
export CUDA_VISIBLE_DEVICES=2

nsys profile -t cuda,nvtx,cublas -c cudaProfilerApi -o $REPORT --force-overwrite true bash benchmarks/basic.sh
