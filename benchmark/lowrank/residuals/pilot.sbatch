#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm101-pilot-1026
#SBATCH --comment=20261026-hdm101-study-v1-pilot
#SBATCH --output=pilot-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source environment.sh
mkdir -p results/pilot
if [[ ! -f inputs/deficient-0/meta.json ]]; then
  srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" common.py inputs pilot > results/pilot/generation.log 2>&1
fi
printf '%s\n' small-0 medium-0 responses-0 wide-0 ill-0 deficient-0 > pilot-tasks.txt
"$HDM_PARALLEL" -j2 --halt soon,fail=1 --joblog results/pilot/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash worker.sh {} pilot' :::: pilot-tasks.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" validate.py "$PWD" pilot
