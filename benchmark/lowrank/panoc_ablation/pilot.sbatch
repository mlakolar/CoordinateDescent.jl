#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm103-ablation-pilot
#SBATCH --comment=20261026-hdm103-controlled-ablation-pilot-v1
#SBATCH --output=pilot-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PKG_OFFLINE=true
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/pilot-v1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$JULIA" --startup-file=no --project=env-baseline-1.13 code/prepare.jl > results/pilot-v1/environment.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" code/generate.py
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
status=0
printf '%s\n' small medium responses wide ill deficient | "$HDM_PARALLEL" -j4 --joblog results/pilot-v1/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash code/worker.sh {}' || status=1
"$PYTHON" code/check.py || status=1
exit "$status"
