#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=cd80-tolerances
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=tolerances-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
mkdir -p reference-loose
parallel --jobs 3 --halt soon,fail=1 --joblog reference-loose/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=1800M bash harness/worker.sh {} R loose reference-loose > reference-loose/worker-{}.log 2>&1' ::: 8201 8202 8203
