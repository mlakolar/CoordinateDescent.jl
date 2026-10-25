#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm102-pilot-1026
#SBATCH --comment=20261026-hdm102-setup-pilot-v1
#SBATCH --output=pilot-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue102/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/pilot
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" common.py inputs pilot > results/pilot/generation.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/env-baseline-1.13 prepare.jl inputs > results/pilot/preparation.log 2>&1
status=0
printf '%s\n' small-0 medium-0 responses-0 wide-0 ill-0 deficient-0 | "$HDM_PARALLEL" -j4 --joblog results/pilot/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash worker.sh {}' || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" check_results.py "$PWD" || status=1
exit "$status"
