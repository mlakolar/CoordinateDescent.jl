#!/bin/bash
set -euo pipefail
cd "$QUANTILE_RUN_ROOT"
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == compatibility ]]; then
 cd ..
 /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --project=compat-env -e 'using HighDimensionalModels,ProximalSolvers,ProximalBase,Test; @assert !ProximalSolvers._hasconjugate(ProximalBase.ConvolutionCheck(.5,.2));include("HighDimensionalModels.jl/test/quantile.jl")'
 exit
fi
module purge
module load r/4.5.3
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --project=../ltsenv worker.jl "$1" "$2"
