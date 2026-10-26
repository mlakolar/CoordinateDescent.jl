#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm107-stat-main
#SBATCH --comment=20261026-hdm107-stat-main-v4
#SBATCH --output=stat-main-v4-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 apptainer parallel/20240522
export HDM_BASE="$PWD" HDM_CODE="$PWD/study-v4-code" HDM_STUDY="$PWD/stat-main-v4" HDM_GATE=main
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v4/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
mkdir -p "$HDM_STUDY/logs"
"$HDM_PARALLEL" --jobs 7 --halt soon,fail=1 --joblog "$HDM_STUDY/generate-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" generate main {} 144 > "$HDM_STUDY/logs/generate-{}.log" 2>&1' ::: $(seq 1 7)
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_STUDY/fit-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" {1} {2} {3} > "$HDM_STUDY/logs/{1}-{2}-{3}.log" 2>&1' ::: native julia R ::: $(seq 1 16) ::: $(seq 1 7)
"$HDM_PARALLEL" --jobs 16 --halt soon,fail=1 --joblog "$HDM_STUDY/check-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" check {1} {2} > "$HDM_STUDY/logs/check-{1}-{2}.log" 2>&1' ::: $(seq 1 16) ::: $(seq 1 7)
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_CODE/study_summary.py" > "$HDM_STUDY/logs/summary.log" 2>&1
echo ALL_STATISTICAL_MAIN_STAGES_FINISHED
