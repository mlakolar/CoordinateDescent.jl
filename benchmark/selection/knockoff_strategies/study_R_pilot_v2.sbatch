#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hdm106-R-pilot
#SBATCH --comment=20261026-hdm106-R-comparator-pilot-v2
#SBATCH --output=R-pilot-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
export HDM_KNOCKOFF_STUDY="$PWD/stat-pilot-v1" HDM_KNOCKOFF_CODE="$PWD/R-pilot-v2-code"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
# Complete only missing old-control checks before the new fit attempt.
"$HDM_PARALLEL" --jobs 4 --halt never --joblog "$HDM_KNOCKOFF_STUDY/R-v1-remaining-check-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/R-v1-remaining-check-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$PWD/R-pilot-v1-code/study_R_check.py" ::: {5..16} ::: 0 || original_check_status=$?
export HDM_R_PREFIX=R-v2
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_STUDY/R-v2-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/R-v2-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_KNOCKOFF_CODE/study_R_worker.sh" ::: {1..16} ::: 0
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_KNOCKOFF_STUDY/R-v2-check-joblog.tsv" --results "$HDM_KNOCKOFF_STUDY/R-v2-check-logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_KNOCKOFF_CODE/study_R_check.py" ::: {1..16} ::: 0
