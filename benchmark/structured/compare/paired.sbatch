#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=9G
#SBATCH --time=00:35:00
#SBATCH --job-name=cd80-paired
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=paired-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
export COMPARE_TIGHT=true
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_PROJECT_ENV="$old/env-release"
mkdir -p baseline-matched optimized-matched optimized-profile
parallel --jobs 3 --halt soon,fail=1 --joblog baseline-matched/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=2800M bash harness/worker.sh {} Julia all baseline-matched > baseline-matched/worker-{}.log 2>&1' ::: 8201 8202 8203
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release"
parallel --jobs 3 --halt soon,fail=1 --joblog optimized-matched/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=2800M bash harness/worker.sh {} Julia all optimized-matched > optimized-matched/worker-{}.log 2>&1' ::: 8201 8202 8203
parallel --jobs 3 --halt soon,fail=1 --joblog optimized-profile/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=2800M bash harness/worker.sh {} Julia profile optimized-profile > optimized-profile/worker-{}.log 2>&1' ::: 8201 8202 8203
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
for output in baseline-matched optimized-matched; do
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl inputs "$output" baseline-full reference-special reference-loose "$output"
done
