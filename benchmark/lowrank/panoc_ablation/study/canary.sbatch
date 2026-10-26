#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=hdm103-study-canary
#SBATCH --comment=20261026-hdm103-study-canary-v1
#SBATCH --output=study-canary-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/study-canary-v1
status=0
for case in medium wide;do
 for method in baseline candidate apg po95 py_fista py_admm pb95 guarded finalreuse guarded_final quadtrait pb99_safe cadence1 cadence20 coeffonly;do printf '%s 0 %s\n' "$case" "$method";done
done | "$HDM_PARALLEL" --colsep ' ' -j8 --joblog results/study-canary-v1/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash study-v1-code/worker.sh {1} {2} {3}' || status=1
"$PYTHON" study-v1-code/summarize.py results/study-canary-v1 --canary || status=1
exit "$status"
