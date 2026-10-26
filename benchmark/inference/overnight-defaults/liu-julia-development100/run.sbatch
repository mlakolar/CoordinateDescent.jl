#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10
#SBATCH --job-name=cd77-liu-julia-development100-20261026
#SBATCH --output=liu-julia-development100-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/source-v1
module load gcc/13.3.0 parallel/20240522
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
parallel --joblog liu-julia-development100-joblog.tsv -j8 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M python3 measure.py resources/liu-julia-development100-{1}.txt /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=../env-hc3-v2 ../liu-julia-development100-v1/study.jl {1} ::: 1 2 3 4 5 6 7 8
python3 ../liu-julia-development100-v1/analyze.py
cd ..
tar czf liu-julia-development100-final-v1.tar.gz source-v1/reports/liu-julia-development100 source-v1/resources/liu-julia-development100-* source-v1/liu-julia-development100-joblog.tsv
