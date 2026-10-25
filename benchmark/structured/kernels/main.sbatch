#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=hdm98-main-1026
#SBATCH --comment=20261026-hdm98-stencil-main-v1
#SBATCH --output=main-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue98/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
mkdir -p results/main tmp
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no generate.jl inputs main > results/main/generation.log 2>&1
for rep in 1 2 3; do
  for case in small irregular linear100k quadratic100k tvpath linearpath quadraticpath linear1m quadratic1m; do printf '%s-%s\n' "$case" "$rep"; done
done > main-tasks.txt
status=0
"$HDM_PARALLEL" -j4 --joblog results/main/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash main-worker.sh {} main' :::: main-tasks.txt || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python check_results.py "$PWD" main || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python summarize.py "$PWD" || status=1
exit "$status"
