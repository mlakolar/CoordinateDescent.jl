#!/bin/bash
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module purge
module load r/4.5.3 gurobi/12.0.3
export R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 BENCH_GUROBI=1
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
apptainer exec --cleanenv --env R_LIBS_USER="$R_LIBS_USER",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript rbench.R "$1" "$2" rbench-results
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. rbench.jl "$1" "$2" rbench-results
