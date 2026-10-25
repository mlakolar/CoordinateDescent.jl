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
module load gcc/13.3.0 apptainer parallel/20240522
export RESULTS_DIR=pilot-v3 CASES=1,4,8,9,10,11
mkdir -p "$RUN/$RESULTS_DIR"
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
# Reuse the already generated shared inputs and successful pilot backends.
for engine in R; do
 python3 "$RUN/source/benchmark/selection/sparse/measure.py" "$RUN/$RESULTS_DIR/rss-$engine.json" bash "$RUN/source/benchmark/selection/sparse/worker.sh" "$engine" 1 timing > "$RUN/$RESULTS_DIR/$engine.log" 2>&1
done
"$JULIA" --startup-file=no --project="$RUN/env" "$RUN/source/benchmark/selection/sparse/audit.jl" 1 > "$RUN/$RESULTS_DIR/audit.log" 2>&1
