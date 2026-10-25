#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=cd80-million
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=million-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release" COMPARE_TIGHT=true COMPARE_INPUT=million/inputs
mkdir -p million/baseline-full million/optimized-selected
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/million.R million/inputs
parallel --jobs 3 --halt soon,fail=1 --joblog million/optimized-selected/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness/worker.sh {} Julia million million/optimized-selected > million/optimized-selected/worker-{}.log 2>&1' ::: 8201 8202 8203
parallel --jobs 3 --halt soon,fail=1 --joblog million/baseline-full/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness/worker.sh {} R million million/baseline-full > million/baseline-full/worker-{}.log 2>&1' ::: 8201 8202 8203
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl million/inputs million/optimized-selected million/baseline-full million/optimized-selected
python3 harness/summarize.py million million/final-analysis
