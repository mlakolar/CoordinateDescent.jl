#!/bin/bash
set -euo pipefail
engine=$1
rep=$2
mode=$3
id=${4:-}
source_dir="$RUN/source/benchmark/selection/sparse"
if [[ $engine == R ]]; then
  export APPTAINERENV_CASES="${CASES:-1,2,3,4,5,6,7,8,9,10,11}"
  apptainer exec --cleanenv --env "RUN=$RUN,RESULTS_DIR=${RESULTS_DIR:-results},OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" \
    /apps/containers/r/4.5/r-4.5.3.sif Rscript "$source_dir/run.R" "$rep" "$mode" ${id:+"$id"}
else
  /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/${ENV_DIR:-env}" "$source_dir/run.jl" "$engine" "$rep" "$mode" ${id:+"$id"}
fi
