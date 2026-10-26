#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=hdm106-R-runtime-canary
#SBATCH --comment=20261026-hdm106-R-runtime-canary-v2
#SBATCH --output=R-runtime-canary-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDM_KNOCKOFF_RUNTIME="$PWD/runtime-pilot-v2" HDM_KNOCKOFF_CODE="$PWD/runtime-R-canary-v2-code"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
export HDM_R_RUNTIME_PREFIX=R-v2
"$HDM_PARALLEL" --jobs 1 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_RUNTIME/R-v2-joblog.tsv" --results "$HDM_KNOCKOFF_RUNTIME/R-v2-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_KNOCKOFF_CODE/runtime_R_worker.sh" ::: 5 ::: 0
