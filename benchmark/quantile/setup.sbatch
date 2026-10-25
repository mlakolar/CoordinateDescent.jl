#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-setup-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=setup-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-quantile
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
for f in CoordinateDescent.jl ProximalBase.jl ProximalSolvers.jl; do tar -xf "$f.tar"; done
tar -xf harness-v1.tar
"$J" --project=. -e 'using Pkg; Pkg.develop([PackageSpec(path="source/ProximalBase.jl"),PackageSpec(path="source/ProximalSolvers.jl"),PackageSpec(path="source/CoordinateDescent.jl")]); Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
"$J" --project=. validate.jl > validation.log 2>&1
"$J" --project=. worker.jl small 101 2 pilot pilot > pilot.log 2>&1
