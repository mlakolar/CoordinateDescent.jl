#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
printf '1\n' | parallel --jobs 1 --halt soon,fail=1 --joblog "$RUN/gaussian-canary-joblog.tsv" \
  "srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash '$RUN/source/benchmark/selection/compare/gaussian_worker.sh' {} > '$RUN/gaussian-{}.log' 2>&1"
seq 2 7 | parallel --jobs 2 --halt soon,fail=1 --joblog "$RUN/gaussian-joblog.tsv" \
  "srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash '$RUN/source/benchmark/selection/compare/gaussian_worker.sh' {} > '$RUN/gaussian-{}.log' 2>&1"
python3 "$RUN/source/benchmark/selection/compare/gaussian_summary.py"
