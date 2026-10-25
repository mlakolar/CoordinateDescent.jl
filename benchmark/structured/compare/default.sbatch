#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=cd80-default
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=default-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release" COMPARE_TIGHT=false
mkdir -p optimized-default
parallel --jobs 3 --halt soon,fail=1 --joblog optimized-default/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=1800M bash harness/worker.sh {} Julia all optimized-default > optimized-default/worker-{}.log 2>&1' ::: 8201 8202 8203
