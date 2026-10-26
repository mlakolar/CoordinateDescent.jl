#!/bin/bash
set -euo pipefail
variant=$1
case=$2
rep=$3
if [[ $variant == candidate ]];then
 environment="$PWD/env-optional3-1.10"
else
 environment="$PWD/env-runtime-baseline-1.10"
fi
/usr/bin/time -v -o "$HDM_KNOCKOFF_RUNTIME/resource-$variant-$case-$rep.txt" /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$environment" "$HDM_KNOCKOFF_CODE/runtime_worker.jl" "$variant" "$case" "$rep"
