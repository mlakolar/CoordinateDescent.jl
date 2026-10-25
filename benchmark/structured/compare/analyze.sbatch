#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd80-analysis
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=analysis-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
for output in baseline-matched optimized-matched optimized-default; do
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-opt-v3-release harness/check.jl inputs "$output" baseline-full reference-special reference-loose "$output"
done
python3 harness/summarize.py . final-analysis
