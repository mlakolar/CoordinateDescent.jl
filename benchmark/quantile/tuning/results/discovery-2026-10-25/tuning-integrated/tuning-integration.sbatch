#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=cd78-tuning-integration-20261025
#SBATCH --output=tuning-integration-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-api/tuning-integrated
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --depwarn=error --project=ltsenv -e 'using HighDimensionalModels,Test;@assert isdefined(HighDimensionalModels,:CVQuantileLassoPath);@assert isdefined(HighDimensionalModels,:fused_lasso);include("HighDimensionalModels.jl/test/quantile.jl");include("HighDimensionalModels.jl/test/quantile_tuning.jl")' > integration-tests.log 2>&1
