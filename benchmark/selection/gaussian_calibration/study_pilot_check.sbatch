#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --job-name=hdm107-pilot-check
#SBATCH --comment=20261026-hdm107-stat-pilot-check-v2
#SBATCH --output=stat-pilot-check-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HDM_BASE="$PWD" HDM_CODE="$PWD/stat-check-v2-code" HDM_STUDY="$PWD/stat-pilot-v1" HDM_GATE=pilot HDM_CHECK_DIR=checks-v2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
"$HDM_PARALLEL" --jobs 8 --halt soon,fail=1 --joblog "$HDM_STUDY/check-v2-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" check {} 0 > "$HDM_STUDY/logs/check-v2-{}.log" 2>&1' ::: $(seq 1 16)
echo ALL_PILOT_CHECKS_PASS_NO_REFITS
