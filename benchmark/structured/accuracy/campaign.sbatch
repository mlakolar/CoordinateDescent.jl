#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:13:00
#SBATCH --job-name=cd97-holdout
#SBATCH --comment=20261025-cd97-defaults
#SBATCH --output=holdout-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
mkdir -p holdout
parallel --jobs 3 --halt soon,fail=1 --joblog holdout/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness/worker.sh {} > holdout/{}.log 2>&1' ::: 8501 8502 8503
