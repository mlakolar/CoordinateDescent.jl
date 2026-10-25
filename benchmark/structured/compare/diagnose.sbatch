#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd80-diagnose
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=diagnose-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
apptainer exec --cleanenv --env R_LIBS_USER="$old/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 \
  /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/diagnose.R inputs graph-diagnosis
