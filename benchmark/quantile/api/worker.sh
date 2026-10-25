#!/bin/bash
set -euo pipefail
cd "$QUANTILE_RUN_ROOT"
module purge
module load r/4.5.3 gurobi/12.0.3
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=1
R=(apptainer exec --cleanenv --env R_LIBS_USER="$R_LIBS_USER",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript)
"${R[@]}" competitors.R generate "$1" "$2"
timeout 1800 /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. worker.jl "$1" "$2"
