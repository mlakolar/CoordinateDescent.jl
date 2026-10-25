#!/bin/bash
set -euo pipefail
stage=$1
rep=$2
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
source_dir="$BASE_RUN/source/benchmark/selection/compare"
julia_run() { "$JULIA" --startup-file=no --project="$BASE_RUN/env" "$source_dir/run.jl" "$1" "$rep"; }
r_run() {
  apptainer exec --cleanenv --env "RUN=$RUN,R_SOLVERS=${R_SOLVERS:-auto},OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" \
    /apps/containers/r/4.5/r-4.5.3.sif Rscript "$source_dir/run.R" "$1" "$rep"
}
if [[ $stage == canary ]]; then
  julia_run generate
  julia_run benchmark
  r_run benchmark
  "$JULIA" --startup-file=no --project="$BASE_RUN/env" "$source_dir/audit.jl" "$rep"
  julia_run mc
  r_run mc
elif [[ $stage == benchmark ]]; then
  julia_run benchmark
  r_run benchmark
  "$JULIA" --startup-file=no --project="$BASE_RUN/env" "$source_dir/audit.jl" "$rep"
elif [[ $stage == mc ]]; then
  julia_run generate
  julia_run mc
  r_run mc
else
  exit 2
fi
echo "WORKER_DONE $stage $rep"
