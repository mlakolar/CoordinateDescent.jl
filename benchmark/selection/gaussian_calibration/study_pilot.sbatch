#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=hdm107-stat-pilot
#SBATCH --comment=20261026-hdm107-stat-pilot-v1
#SBATCH --output=stat-pilot-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 apptainer parallel/20240522
export HDM_BASE="$PWD" HDM_CODE="$PWD/stat-v1-code" HDM_STUDY="$PWD/stat-pilot-v1" HDM_GATE=pilot
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
mkdir -p "$HDM_STUDY/logs"
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" generate pilot 0 2 > "$HDM_STUDY/logs/generate.log" 2>&1
"$HDM_PARALLEL" --jobs 8 --halt soon,fail=1 --joblog "$HDM_STUDY/fit-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" {1} {2} 0 > "$HDM_STUDY/logs/{1}-{2}.log" 2>&1' ::: native julia R ::: $(seq 1 16)
"$HDM_PARALLEL" --jobs 8 --halt soon,fail=1 --joblog "$HDM_STUDY/check-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/study_worker.sh" check {} 0 > "$HDM_STUDY/logs/check-{}.log" 2>&1' ::: $(seq 1 16)
echo ALL_PILOT_STAGES_PASS
