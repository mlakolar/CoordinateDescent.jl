#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=hdm107-runtime-canary
#SBATCH --comment=20261026-hdm107-runtime-canary-v4
#SBATCH --output=runtime-canary-v4-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HDM_BASE="$PWD" HDM_CODE="$PWD/study-v4-code" HDM_STUDY="$PWD/runtime-pilot-v4" HDM_GATE=pilot HDM_ALLOC_SAMPLE_RATE=0.001
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
test ! -e "$HDM_STUDY"
mkdir -p "$HDM_STUDY/fits" "$HDM_STUDY/logs"
ln -s "$HDM_BASE/runtime-pilot-v1/inputs" "$HDM_STUDY/inputs"
for engine in julia R;do
 for sc in 1 2 3 4;do
  for item in "$HDM_BASE/runtime-pilot-v1/fits/$engine-sc$sc-shard0"*;do
   ln -s "$item" "$HDM_STUDY/fits/$(basename "$item")"
  done
 done
done
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_STUDY/fit-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M python3 "$HDM_CODE/resource_wrapper.py" "$HDM_STUDY/logs/native-{}.resource.json" bash "$HDM_CODE/runtime_worker.sh" native {} 0 > "$HDM_STUDY/logs/native-{}.log" 2>&1' ::: $(seq 1 4)
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_STUDY/check-joblog.tsv" \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash "$HDM_CODE/runtime_worker.sh" check {} 0 > "$HDM_STUDY/logs/check-{}.log" 2>&1' ::: $(seq 1 4)
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_CODE/runtime_summary.py" > "$HDM_STUDY/logs/summary.log" 2>&1
echo ALL_CORRECTED_RUNTIME_CANARY_STAGES_PASS
