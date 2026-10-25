#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --job-name=cd80-summary
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=summary-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 harness/summarize.py . final-analysis
module load apptainer
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/plot.R . final-analysis
