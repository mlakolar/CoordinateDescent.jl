#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --job-name=cd78-tuning-20261025
#SBATCH --output=tuning-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export QUANTILE_RUN_ROOT="$PWD"
module load parallel/20240522
mkdir -p logs results
parallel --jobs 2 --joblog results/parallel.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G --cpu-bind=cores bash worker.sh {} > logs/{}.log 2>&1' ::: tests reference
