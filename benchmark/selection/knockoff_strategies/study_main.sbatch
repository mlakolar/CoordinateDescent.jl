#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm106-stat-main
#SBATCH --comment=20261026-hdm106-stat-main-v1
#SBATCH --output=stat-main-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_KNOCKOFF_STUDY="$PWD/stat-main-v1"
test ! -e "$HDM_KNOCKOFF_STUDY"
mkdir "$HDM_KNOCKOFF_STUDY"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_STUDY/generation-fit-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/fit-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$PWD/env-optional3-1.10" study-main-v1-code/study_worker.jl ::: {1..16} ::: {1..7} ::: 144
"$HDM_PARALLEL" --jobs 16 --halt never --joblog "$HDM_KNOCKOFF_STUDY/check-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/check-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python study-main-v1-code/study_check.py ::: {1..16} ::: {1..7} ::: 144
# Comparator fits are a 336-input subset, with the same saved augmented designs.
module load apptainer
export HDM_KNOCKOFF_CODE="$PWD/study-main-v1-code" HDM_R_PREFIX=R
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_STUDY/R-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/R-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_KNOCKOFF_CODE/study_R_worker.sh" ::: {1..16} ::: {1..7}
comparator_status=0
"$HDM_PARALLEL" --jobs 16 --halt never --joblog "$HDM_KNOCKOFF_STUDY/R-check-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/R-check-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_KNOCKOFF_CODE/study_R_check.py" ::: {1..16} ::: {1..7} || comparator_status=$?
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_KNOCKOFF_CODE/study_report.py" 1,2,3,4,5,6,7 144
printf '%s\n' "$comparator_status" > "$HDM_KNOCKOFF_STUDY/R-check-exit-status.txt"
