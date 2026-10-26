#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hdm103-public-recovery
#SBATCH --comment=20261026-hdm103-duplicate-setup-recovery-v2
#SBATCH --output=public-recovery-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
mkdir -p results/pilot-public-recovery-v2
for case in small medium responses wide ill deficient;do
 for config in native apg;do printf '%s %s\n' "$case" "$config";done
done | "$HDM_PARALLEL" --colsep ' ' -j4 --joblog results/pilot-public-recovery-v2/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash code/recover-public-worker.sh {1} {2}'
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python code/recovery_report.py
