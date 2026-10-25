#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-R-20261025
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=rsetup-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-quantile
module load r/4.5.3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p Rlib
export R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3"
Rscript -e 'options(repos=c(CRAN="https://cloud.r-project.org")); need=setdiff(c("quantreg","conquer","rqPen"),rownames(installed.packages())); if(length(need)) install.packages(need,lib="Rlib",Ncpus=1); print(sapply(c("quantreg","conquer","rqPen"),packageVersion)); writeLines(capture.output(sessionInfo()),"R-session.txt")'
