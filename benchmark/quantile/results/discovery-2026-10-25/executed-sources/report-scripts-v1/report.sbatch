#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-report-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=report-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=0
for pair in 'main-results summary' 'pilot-v2 pilot-summary' 'admm-results admm-summary' 'rbench-results rbench-summary'; do
  read -r data dest <<< "$pair"
  if [ ! -f "$dest/coverage.json" ]; then python3 summarize.py "$data" "$dest"; fi
done
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. approximation.jl rbench-results approximation.csv
python3 report_inputs.py .
