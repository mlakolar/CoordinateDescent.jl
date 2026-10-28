#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm101-main-1026
#SBATCH --comment=20261026-hdm101-study-v1-main
#SBATCH --output=main-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source environment.sh
mkdir -p results/main
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" common.py inputs main > results/main/generation.log 2>&1
for rep in {1..20}; do
  for case in small medium responses wide ill deficient; do printf '%s-%s\n' "$case" "$rep"; done
done > main-tasks.txt
status=0
"$HDM_PARALLEL" -j8 --joblog results/main/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash worker.sh {} main' :::: main-tasks.txt || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" validate.py "$PWD" main || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" summarize.py "$PWD" main || status=1
exit "$status"
