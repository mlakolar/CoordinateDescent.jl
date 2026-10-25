#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cd78-api-focused-20261025
#SBATCH --output=focused-%j.out
set -euo pipefail
export QUANTILE_RUN_ROOT="${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$QUANTILE_RUN_ROOT"
module load parallel/20240522
mkdir -p inputs results logs
worker_status=0
if ! parallel --colsep '\t' --jobs 2 --joblog results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G --cpu-bind=cores timeout 1800 bash worker.sh {1} {2} > logs/{1}-{2}.log 2>&1' :::: tasks.tsv; then worker_status=1; fi
python3 summarize.py .

exit "$worker_status"
