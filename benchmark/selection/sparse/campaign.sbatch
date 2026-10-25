#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
source_dir="$RUN/source/benchmark/selection/sparse"
export ENV_DIR=env335
mkdir -p "$RUN/results" "$RUN/memory" "$RUN/metadata"
cp "$RUN/$ENV_DIR/Project.toml" "$RUN/$ENV_DIR/Manifest.toml" "$RUN/metadata/"
export RESULTS_DIR=results CASES=1,2,3,4,5,6,7,8,9,10,11
# Coordinated PS engine LTS tests, outside all benchmark timings.
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/$ENV_DIR" /home1/mkolar/juliaPkg/runs/20261025T1210Z-ps-slope-extract/source335/test/slope.jl > "$RUN/engine-tests-110.log" 2>&1
cd "$RUN/source"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/$ENV_DIR" -e 'using CoordinateDescent,Test; include("test/selection.jl"); include("test/slope_sparse.jl"); include("test/selection_golden.jl")' > "$RUN/final335-tests-110.log" 2>&1
# Serialize timing processes; rotate backend order between replications.
python3 - <<'PY' > "$RUN/tasks.tsv"
engines=['native','baseline','SLOPE.jl','R']
for rep in range(1,8):
 order=engines[(rep-1)%4:]+engines[:(rep-1)%4]
 if rep%2==0: order=order[::-1]
 for engine in order: print(engine,rep,sep='\t')
PY
parallel --colsep '\t' --jobs 1 --halt soon,fail=1 --joblog "$RUN/timing-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=4G python3 '$source_dir/measure.py' '$RUN/results/process-rss-{1}-{2}.json' bash '$source_dir/worker.sh' {1} {2} timing > '$RUN/results/worker-{1}-{2}.log' 2>&1" :::: "$RUN/tasks.tsv"
# Peak RSS needs a fresh process for each size, rather than a cumulative
# high-water mark after unrelated cases. Keep these separate from timing.
export RESULTS_DIR=memory
python3 - <<'PY' > "$RUN/memory-tasks.tsv"
for rep in range(1,8):
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if case==11 and engine=='baseline':continue
   print(engine,rep,case,sep='\t')
PY
parallel --colsep '\t' --jobs 4 --halt soon,fail=1 --joblog "$RUN/memory-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=4G python3 '$source_dir/measure.py' '$RUN/memory/rss-{1}-{2}-{3}.json' bash '$source_dir/worker.sh' {1} {2} memory {3} > '$RUN/memory/worker-{1}-{2}-{3}.log' 2>&1" :::: "$RUN/memory-tasks.tsv"
export RESULTS_DIR=results
seq 1 7 | parallel --jobs 1 --halt soon,fail=1 --joblog "$RUN/audit-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=4G bash '$source_dir/audit_worker.sh' {} > '$RUN/results/audit-{}.log' 2>&1"
python3 "$source_dir/summarize.py"
python3 "$source_dir/profile_summary.py"
echo CAMPAIGN_DONE
