#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd78-tuning-diagnostic-20261025
#SBATCH --output=tuning-diagnostic-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export QUANTILE_RUN_ROOT="$PWD"
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module purge
module load r/4.5.3
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
J=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
mkdir -p tied regular
cd tied
apptainer exec --cleanenv --env "R_LIBS_USER=$R_LIBS_USER,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1" /apps/containers/r/4.5/r-4.5.3.sif Rscript ../reference.R > R.log 2>&1
"$J" --project="$QUANTILE_RUN_ROOT/ltsenv" ../diagnose.jl > diagnosis.log 2>&1
cd ../regular
apptainer exec --cleanenv --env "R_LIBS_USER=$R_LIBS_USER,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1" /apps/containers/r/4.5/r-4.5.3.sif Rscript ../reference.R regular > R.log 2>&1
"$J" --depwarn=error --project="$QUANTILE_RUN_ROOT/ltsenv" ../validate.jl > validation.log 2>&1
