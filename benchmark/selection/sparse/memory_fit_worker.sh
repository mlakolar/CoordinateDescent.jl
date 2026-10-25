#!/bin/bash
set -euo pipefail
engine=$1 rep=$2 id=$3
if [[ $engine == R ]]; then
  apptainer exec --cleanenv --env "RUN=$RUN,RESULTS_DIR=memory,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" \
    /apps/containers/r/4.5/r-4.5.3.sif Rscript "$FROZEN_RUN/source/benchmark/selection/sparse/run.R" "$rep" memory "$id"
else
  /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/env" "$RUN/scripts/memory_fit_only.jl" "$engine" "$rep" "$id"
fi
