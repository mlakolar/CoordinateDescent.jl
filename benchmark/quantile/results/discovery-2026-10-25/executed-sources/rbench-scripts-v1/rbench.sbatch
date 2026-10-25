#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-Rbench-20261025
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=rbench-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-quantile
module load parallel/20240522
mkdir -p rbench-results rbench-logs
parallel --colsep '\t' --jobs 2 --joblog rbench-results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores timeout 320 bash rbench_worker.sh {1} {2} > rbench-logs/{1}-{2}.log 2>&1' :::: rbench-tasks.tsv
python3 rbench_summarize.py rbench-results rbench-summary
