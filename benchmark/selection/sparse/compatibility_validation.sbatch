#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:14:00
set -euo pipefail
export BASE_RUN="$RUN"
export RUN="$BASE_RUN/compat" ENV_DIR=env
export JULIA_DEPOT_PATH="$RUN/depot:$BASE_RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source_dir="$RUN/source/benchmark/selection/sparse"
mkdir -p "$RUN/metadata"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no "$source_dir/compatibility_setup.jl" > "$RUN/setup-110.log" 2>&1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no "$source_dir/compatibility_tests.jl" > "$RUN/full-tests-113.log" 2>&1
cd "$RUN/source"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/env" -e 'using CoordinateDescent,Test; include("test/selection.jl"); include("test/slope_sparse.jl"); include("test/selection_golden.jl"); include(joinpath(ENV["RUN"],"ps-source","test","slope.jl"))' > "$RUN/targeted-tests-110.log" 2>&1
cp "$RUN/env/Project.toml" "$RUN/env/Manifest.toml" "$RUN/metadata/"
echo CURRENT_COMPATIBILITY_VALIDATION_DONE
