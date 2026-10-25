#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:14:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SPARSE_PS_SOURCE="$RUN/ps-source335" SPARSE_TEST_LABEL=test-source-335 SPARSE_TEST_ENV=testenv-335
cd "$RUN/source"
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no benchmark/selection/sparse/fulltests.jl > "$RUN/fulltests-335-113.log" 2>&1
