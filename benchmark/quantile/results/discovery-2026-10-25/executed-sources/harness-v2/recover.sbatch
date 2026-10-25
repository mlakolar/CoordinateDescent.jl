#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-canary2-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=canary2-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-quantile
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gurobi/12.0.3
export GUROBI_JL_USE_GUROBI_JLL=false
J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
tar -xf harness-v2.tar
"$J" --project=. -e 'using Pkg; Pkg.add("Gurobi"); Pkg.build("Gurobi"); using Gurobi; Gurobi.Env(output_flag=0)' > gurobi-setup.log 2>&1
export BENCH_GUROBI=1
"$J" --project=. validate.jl > validation-v2.log 2>&1
"$J" --project=. worker.jl small 101 2 pilot pilot-v2 > pilot-v2.log 2>&1
