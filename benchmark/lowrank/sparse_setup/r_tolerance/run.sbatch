#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=hdm102-rtol-1026
#SBATCH --comment=20261026-hdm102-R-tolerance-fairness-v1
#SBATCH --output=r-tol-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDM_HARNESS=r-tol-v1
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
mkdir -p results/main-r-tol-v1
status=0
for rep in $(seq 1 20);do for case in small medium responses wide ill deficient;do printf '%s-%s\n' "$case" "$rep";done;done | "$HDM_PARALLEL" -j4 --joblog results/main-r-tol-v1/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash r-tol-v1/worker.sh {}' || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" r-tol-v1/full_check.py "$PWD" main-r-tol-v1 || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" r-tol-v1/path_check.py "$PWD" main-r-tol-v1 || status=1
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" r-tol-v1/full_summary.py "$PWD" main-r-tol-v1 || status=1
exit "$status"
