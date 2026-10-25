#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:12:00
#SBATCH --job-name=cd78-api-lts-renamed-20261025
#SBATCH --output=lts-renamed-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-api
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p integration-source/ltsenv
cd integration-source/ltsenv
J=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
"$J" --project=. -e 'using Pkg; Pkg.develop([PackageSpec(path="../HighDimensionalModels.jl"),PackageSpec(path="../ProximalOPT.jl"),PackageSpec(path="../ProximalBase.jl")]); Pkg.add(PackageSpec(name="HiGHS",version="1.26.1")); Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
"$J" --depwarn=error --project=. -e 'using HighDimensionalModels,Test; include("../HighDimensionalModels.jl/test/quantile.jl")' > ../../api-tests-1.10-renamed.log 2>&1
