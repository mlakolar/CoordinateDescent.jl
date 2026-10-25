#!/bin/bash
#SBATCH --job-name=cd82-repair-examples-b388317
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:03:00
#SBATCH --output=/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/repair-74bc14b/examples-%j.log
set -euo pipefail
export FIX_ROOT=/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/repair-74bc14b
export JULIA_DEPOT_PATH="$FIX_ROOT/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/renamed/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --exclusive --exact -N1 -n1 -c2 --mem=6G /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --check-bounds=yes --project="$FIX_ROOT/env-1.13-clean" -e 'using Test; @testset "Solver examples and Aqua after staging repair" begin; include(joinpath(ENV["FIX_ROOT"],"ProximalOPT-clean.jl","test","examples.jl")); include(joinpath(ENV["FIX_ROOT"],"ProximalOPT-clean.jl","test","aqua.jl")); end; println("SOLVER_REMAINING_PASS")' > "$FIX_ROOT/solver-remaining.log" 2>&1
echo SOLVER_ALL_FILES_VALIDATED
