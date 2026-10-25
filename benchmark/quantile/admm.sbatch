#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-admm-20261025
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=admm-%j.out
set -euo pipefail
export QUANTILE_RUN_ROOT="${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module load parallel/20240522
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=0
export J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
"$J" --project=. admm_study.jl test 0 admm-results > admm-validation.log 2>&1
mkdir -p admm-logs
parallel --colsep '\t' --jobs 2 --joblog admm-results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores timeout 220 "$J" --project=. admm_study.jl {1} {2} admm-results > admm-logs/{1}-{2}.log 2>&1' :::: admm-tasks.tsv
python3 summarize.py admm-results admm-summary
