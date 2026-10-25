#!/bin/bash
set -euo pipefail
seed="$1"; engine="$2"; mode="$3"; output="$4"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
input="${COMPARE_INPUT:-inputs}"
harness="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JULIA_DEPOT_PATH="${COMPARE_DEPOT_PATH:-$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:}"
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
if [ "$engine" = R ]; then
  apptainer exec --cleanenv --env R_LIBS_USER="$old/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,TMPDIR="$old/tmp" \
    /apps/containers/r/4.5/r-4.5.3.sif Rscript "$harness/reference.R" "$seed" "$input" "$output" "$mode"
else
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" "$harness/julia.jl" "$seed" "$input" "$output" "$mode"
fi
