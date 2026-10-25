#!/bin/bash
set -euo pipefail
cd "$QUANTILE_RUN_ROOT"
module purge
module load r/4.5.3
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == regressions ]]; then
 /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --project=ltsenv regressions.jl
 exit
fi
if [[ ! -f inputs/$1-$2-meta.csv ]]; then
 apptainer exec --cleanenv /apps/containers/r/4.5/r-4.5.3.sif Rscript generator.R generate "$1" "$2"
fi
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --project=ltsenv worker.jl "$1" "$2"
