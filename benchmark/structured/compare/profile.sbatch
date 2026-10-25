#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=cd80-profile
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=profile-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
export JULIA_PROJECT_ENV="${JULIA_PROJECT_ENV:-/home1/mkolar/juliaPkg/runs/20261025-cd80-structured/env-release}"
mkdir -p baseline-profile
parallel --jobs 3 --halt soon,fail=1 --joblog baseline-profile/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness/worker.sh {} Julia profile baseline-profile > baseline-profile/worker-{}.log 2>&1' ::: 8201 8202 8203
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl inputs baseline-canary-v2 baseline-canary-v2
