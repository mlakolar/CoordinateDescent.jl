#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=hdm103-study-v1
#SBATCH --comment=20261026-hdm103-replicated-study-v1
#SBATCH --output=study-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/study-v1
# Canary gate runs before any fresh input generation.
"$PYTHON" study-v1-code/summarize.py results/study-canary-v1 --canary
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" study-v1-code/generate.py inputs/study-v1
status=0
for phase in primary controls;do
 "$PYTHON" study-v1-code/tasks.py "$phase" | "$HDM_PARALLEL" --colsep ' ' -j16 --joblog "results/study-v1/$phase-workers.tsv" 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash study-v1-code/worker.sh {1} {2} {3}' || status=1
done
"$PYTHON" study-v1-code/summarize.py results/study-v1 || status=1
exit "$status"
