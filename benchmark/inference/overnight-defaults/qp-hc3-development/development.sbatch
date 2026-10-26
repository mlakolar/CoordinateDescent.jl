#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=20
#SBATCH --job-name=cd77-qphc3-development100-20261026
#SBATCH --output=qphc3-development100-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/source-v1
module load gcc/13.3.0 parallel/20240522
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
parallel --colsep '\t' --joblog qphc3-development100-joblog.tsv -j32 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M python3 measure.py resources/qphc3-development100-{1}-{2}.txt /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=../env-hc3-v2 ../qphc3-v1/study.jl {1} {2} {3} {4} :::: ../qphc3-v1/tasks.tsv
python3 ../qphc3-v1/analyze.py development 100
cd ..
tar czf qphc3-development100-final-v1.tar.gz source-v1/reports/development-qphc3 source-v1/resources/qphc3-development100-* source-v1/qphc3-development100-joblog.tsv
