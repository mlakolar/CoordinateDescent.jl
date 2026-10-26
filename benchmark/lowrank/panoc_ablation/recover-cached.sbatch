#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hdm103-cached-recovery
#SBATCH --comment=20261026-hdm103-cached-label-recovery-v3
#SBATCH --output=cached-recovery-v3-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
mkdir -p results/pilot-cached-recovery-v3
printf '%s\n' small medium responses wide ill deficient | "$HDM_PARALLEL" -j4 --joblog results/pilot-cached-recovery-v3/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash code/recover-cached-worker.sh {}'
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python code/recovery_report_v3.py
