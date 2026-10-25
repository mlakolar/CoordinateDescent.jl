#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=cd80-setup
#SBATCH --comment=20261025-cd80-structured
#SBATCH --output=setup-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p Rlib tmp
module load apptainer
apptainer exec --cleanenv --env R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,TMPDIR="$PWD/tmp" \
 /apps/containers/r/4.5/r-4.5.3.sif Rscript setup.R
curl -fL --retry 2 https://codeload.github.com/glmgen/glmgen/tar.gz/9e6061503d6dd103712a0c07f6be7293c1c58aaf -o glmgen.tar.gz
tar xzf glmgen.tar.gz
apptainer exec --cleanenv --env R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,TMPDIR="$PWD/tmp" \
 /apps/containers/r/4.5/r-4.5.3.sif R CMD INSTALL --library="$PWD/Rlib" glmgen-9e6061503d6dd103712a0c07f6be7293c1c58aaf/R_pkg/glmgen
