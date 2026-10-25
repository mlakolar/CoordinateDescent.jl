#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=hdm102-main-1026
#SBATCH --comment=20261026-hdm102-full-20input-main-v1
#SBATCH --output=full-main-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue102/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
export HDM_HARNESS=main-v1
mkdir -p results/main
# Summarize the passed adapter canary before generating any holdout inputs.
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" main-v1/full_summary.py "$PWD" pilot-adapters-v2
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" main-v1/common.py inputs main
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=../issue99/env-baseline-1.13 main-v1/prepare.jl inputs main
status=0
for rep in $(seq 1 20); do for case in small medium responses wide ill deficient; do printf '%s-%s\n' "$case" "$rep"; done; done | "$HDM_PARALLEL" -j8 --joblog results/main/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash main-v1/full_worker.sh {} main' || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" main-v1/full_check.py "$PWD" main || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" main-v1/path_check.py "$PWD" main || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" main-v1/full_summary.py "$PWD" main || status=1
exit "$status"
