#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm102-fullcal-1026
#SBATCH --comment=20261026-hdm102-full-adapter-calibration-v1
#SBATCH --output=full-calibration-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue102/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/pilot-full-v1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=../issue99/env-baseline-1.13 full-v1/prepare.jl inputs pilot
status=0
printf '%s\n' small-0 medium-0 responses-0 wide-0 ill-0 deficient-0 | "$HDM_PARALLEL" -j6 --joblog results/pilot-full-v1/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash full-v1/full_worker.sh {} pilot-full-v1' || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" full-v1/full_check.py "$PWD" pilot-full-v1 || status=1
exit "$status"
