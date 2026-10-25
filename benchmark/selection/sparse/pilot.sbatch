#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
cd "$RUN/source"
"$JULIA" --startup-file=no --project="$RUN/env" -e 'using CoordinateDescent, Test; include("test/selection.jl"); include("test/slope_sparse.jl"); include("test/selection_golden.jl")' > "$RUN/tests-110.log" 2>&1
if [[ -f benchmark/selection/sparse/generate.jl ]]; then
  "$JULIA" --startup-file=no --project="$RUN/env" benchmark/selection/sparse/generate.jl > "$RUN/generate.log" 2>&1
fi
