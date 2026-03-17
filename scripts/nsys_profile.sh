#!/bin/bash
# Usage: bash nsys_profile.sh [REPORT NAME]

PREFIX="~/nsys_reports/nsys_"
REPORT=${PREFIX}${1}

echo "Creating $REPORT"

nsys profile -t cuda,nvtx,cublas -c cudaProfilerApi -o $REPORT --force-overwrite true bash benchmarks/basic.sh
