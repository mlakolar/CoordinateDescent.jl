#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd80-stopping-holdout-J
#SBATCH --comment=20261025-cd80-stopping
#SBATCH --output=holdout-J-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
python3 harness-holdout/select.py
mkdir -p holdout/julia
parallel --jobs 3 --halt soon,fail=1 --joblog holdout/julia/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness-holdout/holdout-worker.sh {} > holdout/julia/worker-{}.log 2>&1' ::: 8301 8302 8303
