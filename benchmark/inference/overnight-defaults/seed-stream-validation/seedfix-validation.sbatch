#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=30
#SBATCH --job-name=cd77-seedfix-validation-20261026
#SBATCH --output=seedfix-validation-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/source-v1
module load gcc/13.3.0 parallel/20240522 apptainer/1.3.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
parallel --colsep '\t' --joblog seedfix-validation-joblog.tsv -j64 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M python3 measure.py resources/seedfix-validation-{1}-{2}.txt /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=../env-hc3-v2 ../seedfix-gaussian-v1/seedfix-gaussian.jl {1} {2} {3} {4} :::: ../seedfix-gaussian-v1/seedfix-validation-tasks.tsv
python3 ../analyze-gaussian-campaign-v2.py seedfix-validation 500
python3 ../seedfix-gaussian-v1/seedfix-gate.py
