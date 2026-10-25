#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
source_dir="$RUN/source/benchmark/selection/sparse"
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
cd "$RUN/source"
"$JULIA" --startup-file=no --project="$RUN/env" -e 'using HighDimensionalModels,Test; include("test/selection.jl"); include("test/slope_sparse.jl"); include("test/selection_golden.jl")' > "$RUN/final-tests-110.log" 2>&1
export RESULTS_DIR=final-pilot CASES=1,4,8,9,10,11
mkdir -p "$RUN/$RESULTS_DIR"
for engine in native baseline SLOPE.jl R; do
 python3 "$source_dir/measure.py" "$RUN/$RESULTS_DIR/rss-$engine.json" bash "$source_dir/worker.sh" "$engine" 1 timing > "$RUN/$RESULTS_DIR/$engine.log" 2>&1
done
"$JULIA" --startup-file=no --project="$RUN/env" "$source_dir/audit.jl" 1 > "$RUN/$RESULTS_DIR/audit.log" 2>&1
export SPARSE_PS_SOURCE="$RUN/ps-source" SPARSE_TEST_LABEL=test-source-final SPARSE_TEST_ENV=testenv-final
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no "$source_dir/fulltests.jl" > "$RUN/fulltests-final-113.log" 2>&1
