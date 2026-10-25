#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:02:00
#SBATCH --job-name=cd97-final-checks
#SBATCH --comment=20261025-cd97-defaults
#SBATCH --output=final-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
mkdir -p validation-final
parallel --jobs 2 --halt soon,fail=1 --joblog validation-final/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness-final/final-worker.sh {} > validation-final/{}.log 2>&1' ::: release lts
