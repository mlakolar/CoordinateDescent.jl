#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm98-refpilot-1026
#SBATCH --comment=20261026-hdm98-reference-pilot-v2
#SBATCH --output=reference-pilot-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
mkdir -p results/reference-pilot tmp
printf '%s\n' small-0 irregular-0 quadratic100k-0 | "$HDM_PARALLEL" -j2 --joblog results/reference-pilot/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash reference-worker.sh {} reference-pilot pilot'
