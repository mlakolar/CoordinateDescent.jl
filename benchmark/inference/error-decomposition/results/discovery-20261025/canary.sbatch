#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cd77-decomp-canary-20261025
#SBATCH --output=canary-%j.out
set -euo pipefail
cd "/home1/mkolar/juliaPkg/runs/20261025-cd77-decomposition"
module load gcc/13.3.0 parallel/20240522
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
mkdir -p resources
parallel --colsep '\t' --joblog canary-joblog.tsv -j8 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M python3 measure.py resources/canary-{1}-{2}.txt /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261025-cd77-bias-tuning/env-v5 study.jl {1} {2} {3} {4} :::: canary-tasks.tsv
python3 analyze.py canary
