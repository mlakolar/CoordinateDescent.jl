#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm106-runtime-pilot
#SBATCH --comment=20261026-hdm106-runtime-main-v1
#SBATCH --output=runtime-main-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_KNOCKOFF_RUNTIME="$PWD/runtime-main-v1" HDM_KNOCKOFF_CODE="$PWD/runtime-main-v1-code"
test ! -e "$HDM_KNOCKOFF_RUNTIME"
mkdir "$HDM_KNOCKOFF_RUNTIME"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_RUNTIME/joblog.tsv" --results "$HDM_KNOCKOFF_RUNTIME/logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_KNOCKOFF_CODE/runtime_worker.sh" ::: baseline candidate ::: {1..5} ::: {1..7}
"$HDM_PARALLEL" --jobs 7 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_RUNTIME/legacy-field-check-joblog.tsv" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$PWD/env-optional3-1.10" "$HDM_KNOCKOFF_CODE/runtime_check.jl" ::: {1..7}
module load apptainer
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_RUNTIME/R-joblog.tsv" --results "$HDM_KNOCKOFF_RUNTIME/R-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_KNOCKOFF_CODE/runtime_R_worker.sh" ::: {1..5} ::: {1..7}
