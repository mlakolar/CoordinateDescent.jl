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
python3 "$HDM_KNOCKOFF_CODE/resource_wrapper.py" "$HDM_KNOCKOFF_RUNTIME/resource-$variant-$case-$rep.json" /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$environment" "$HDM_KNOCKOFF_CODE/runtime_worker.jl" "$variant" "$case" "$rep"
