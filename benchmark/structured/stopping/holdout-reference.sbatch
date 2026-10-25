#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=cd80-stopping-holdout-R
#SBATCH --comment=20261025-cd80-stopping
#SBATCH --output=holdout-R-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/holdout.R holdout/inputs "$prior"
mkdir -p holdout/reference
parallel --jobs 3 --halt soon,fail=1 --joblog holdout/reference/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M apptainer exec --cleanenv --env R_LIBS_USER='"$old"'/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,TMPDIR='"$old"'/tmp /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/reference.R {} holdout/inputs holdout/reference million > holdout/reference/worker-{}.log 2>&1' ::: 8301 8302 8303
