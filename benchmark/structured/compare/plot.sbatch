#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --job-name=cd80-report-plots
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=plots-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript "${1:-harness}/plot.R" . "${2:-final-report}"
