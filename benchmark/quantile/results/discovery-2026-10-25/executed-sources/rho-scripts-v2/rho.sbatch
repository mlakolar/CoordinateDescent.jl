#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-rho-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --output=rho-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=0
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. rho_study.jl rho-results
python3 summarize.py rho-results rho-summary
