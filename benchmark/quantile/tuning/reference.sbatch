#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd78-tuning-reference-20261025
#SBATCH --output=tuning-reference-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export QUANTILE_RUN_ROOT="$PWD"
mkdir -p logs results
srun --exclusive --exact -N1 -n1 -c1 --mem=8G --cpu-bind=cores bash worker.sh reference > logs/reference.log 2>&1
