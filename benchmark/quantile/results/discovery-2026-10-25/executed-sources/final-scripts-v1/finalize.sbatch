#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-analysis-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=finalize-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-quantile
module purge
module load r/4.5.3
export R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3"
apptainer exec --cleanenv --env R_LIBS_USER="$R_LIBS_USER",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript golden-v2.R
python3 summarize.py pilot-v2 pilot-summary
python3 summarize.py main-results summary
