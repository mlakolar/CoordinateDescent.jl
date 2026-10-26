#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm107-runtime-recover
#SBATCH --comment=20261026-hdm107-runtime-pilot-recovery-v3
#SBATCH --output=runtime-recovery-v3-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 apptainer parallel/20240522
export HDM_BASE="$PWD" HDM_CODE="$PWD/study-v3-code" HDM_STUDY="$PWD/runtime-pilot-v3" HDM_GATE=pilot HDM_RESUME=1 HDM_ALLOC_SAMPLE_RATE=0.001
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" "$HDM_CODE/runtime_recover.py"
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_STUDY/fit-joblog.tsv" \
 'test -f "$HDM_STUDY/fits/native-sc{}-shard0.done" || srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/runtime_worker.sh" native {} 0 > "$HDM_STUDY/logs/native-{}.log" 2>&1' ::: $(seq 1 4)
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_STUDY/check-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/runtime_worker.sh" check {} 0 > "$HDM_STUDY/logs/check-{}.log" 2>&1' ::: $(seq 1 4)
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" "$HDM_CODE/runtime_summary.py" > "$HDM_STUDY/logs/summary.log" 2>&1
export HDM_STUDY="$PWD/stat-pilot-v1" HDM_CHECK_DIR=checks-v2
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" "$HDM_CODE/study_summary.py" > "$HDM_STUDY/logs/summary-v3.log" 2>&1
echo ALL_PILOT_GATES_COMPLETE
