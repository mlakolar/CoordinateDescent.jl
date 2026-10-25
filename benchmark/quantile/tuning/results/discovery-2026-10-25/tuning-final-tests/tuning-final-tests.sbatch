#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=cd78-tuning-final-20261025
#SBATCH --output=tuning-final-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-api/tuning-final-tests
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
"$J" --depwarn=error --project=ltsenv -e 'using HighDimensionalModels,ProximalSolvers,ProximalBase,Test;@assert ProximalSolvers._hasconjugate(ProximalBase.ConvolutionCheck(.5,.2));@assert Threads.nthreads()==2;include("HighDimensionalModels.jl/test/quantile_tuning.jl")' > new-backend.log 2>&1
"$J" --depwarn=error --project=compat-env -e 'using HighDimensionalModels,ProximalSolvers,ProximalBase,Test;@assert !ProximalSolvers._hasconjugate(ProximalBase.ConvolutionCheck(.5,.2));@assert Threads.nthreads()==2;include("HighDimensionalModels.jl/test/quantile_tuning.jl")' > old-backend.log 2>&1
