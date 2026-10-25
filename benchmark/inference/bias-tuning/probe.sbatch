#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=cd77-bias-dev-20261025
#SBATCH --output=probe-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 parallel/20240522
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
parallel --colsep '\t' --joblog probe-joblog.tsv -j8 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-v1 probe-v1.jl {1} {2} {3} {4} :::: probe-tasks.tsv
