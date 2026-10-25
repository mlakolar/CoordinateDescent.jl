#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=epyc-7313
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail
export BASE_RUN="$RUN"
export RUN="$BASE_RUN/compat" ENV_DIR=env
export JULIA_DEPOT_PATH="$RUN/depot:$BASE_RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
source_dir="$RUN/source/benchmark/selection/sparse"
mkdir -p "$RUN/results" "$RUN/memory" "$RUN/metadata"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no "$source_dir/compatibility_setup.jl" > "$RUN/setup-110.log" 2>&1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no "$source_dir/compatibility_tests.jl" > "$RUN/full-tests-113.log" 2>&1
export JULIA_NUM_THREADS=1 RESULTS_DIR=results CASES=1,2,3,4,10,11 CASE_IDS=1,2,3,4,10,11
cd "$RUN/source"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/env" -e 'using CoordinateDescent,Test; include("test/selection.jl"); include("test/slope_sparse.jl"); include("test/selection_golden.jl"); include(joinpath(ENV["RUN"],"ps-source","test","slope.jl"))' > "$RUN/targeted-tests-110.log" 2>&1
cp "$RUN/env/Project.toml" "$RUN/env/Manifest.toml" "$RUN/metadata/"
python3 - <<'PY' > "$RUN/tasks.tsv"
engines=['native','baseline','SLOPE.jl','R']
for rep in range(1,8):
 order=engines[(rep-1)%4:]+engines[:(rep-1)%4]
 if rep%2==0:order=order[::-1]
 for engine in order:print(engine,rep,sep='\t')
PY
parallel --colsep '\t' --jobs 1 --halt soon,fail=1 --joblog "$RUN/timing-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=6G python3 '$source_dir/measure.py' '$RUN/results/process-rss-{1}-{2}.json' bash '$source_dir/worker.sh' {1} {2} timing > '$RUN/results/worker-{1}-{2}.log' 2>&1" :::: "$RUN/tasks.tsv"
export RESULTS_DIR=memory
python3 - <<'PY' > "$RUN/memory-tasks.tsv"
for rep in range(1,8):
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if case==11 and engine=='baseline':continue
   print(engine,rep,case,sep='\t')
PY
parallel --colsep '\t' --jobs 2 --halt soon,fail=1 --joblog "$RUN/memory-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=6G python3 '$source_dir/measure.py' '$RUN/memory/rss-{1}-{2}-{3}.json' bash '$source_dir/worker.sh' {1} {2} memory {3} > '$RUN/memory/worker-{1}-{2}-{3}.log' 2>&1" :::: "$RUN/memory-tasks.tsv"
seq 1 7 | parallel --jobs 1 --halt soon,fail=1 --joblog "$RUN/audit-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash '$source_dir/compatibility_audit.sh' {} > '$RUN/results/audit-{}.log' 2>&1"
export RESULTS_DIR=results
python3 "$source_dir/summarize.py"
python3 "$source_dir/profile_summary.py"
echo COMPATIBILITY_CONFIRMATION_DONE
