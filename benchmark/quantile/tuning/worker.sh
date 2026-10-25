#!/bin/bash
set -euo pipefail
cd "$QUANTILE_RUN_ROOT"
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
if [[ "$1" == tests ]]; then
 "$J" --depwarn=error --project=ltsenv -e 'using HighDimensionalModels,Test;include("HighDimensionalModels.jl/test/quantile.jl");include("HighDimensionalModels.jl/test/quantile_tuning.jl")'
else
 module purge
 module load r/4.5.3
 export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
 apptainer exec --cleanenv --env "R_LIBS_USER=$R_LIBS_USER,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1" /apps/containers/r/4.5/r-4.5.3.sif Rscript reference.R regular
 "$J" --depwarn=error --project=ltsenv validate.jl
fi
