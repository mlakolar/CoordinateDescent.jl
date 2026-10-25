#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd80-special
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=special-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release"
for seed in 8201 8202 8203; do
  bash harness/worker.sh "$seed" R special reference-special
done
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl inputs reference-special baseline-canary-v2 baseline-full optimized-canary reference-special
