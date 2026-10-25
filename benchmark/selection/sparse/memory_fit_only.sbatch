#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:14:00
set -euo pipefail
export FROZEN_RUN="$RUN" RUN="$RUN/rss-fit-only"
export JULIA_DEPOT_PATH="$FROZEN_RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export BENCHMARK_MODULE=CoordinateDescent RESULTS_DIR=memory CASES=4,10,11
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
mkdir -p "$RUN/memory" "$RUN/results"
python3 - <<'PY' > "$RUN/tasks.tsv"
for rep in range(1,8):
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if case==11 and engine=='baseline':continue
   print(engine,rep,case,sep='\t')
PY
parallel --colsep '\t' --jobs 4 --halt soon,fail=1 --joblog "$RUN/memory-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c1 --mem=2G python3 '$FROZEN_RUN/source/benchmark/selection/sparse/measure.py' '$RUN/memory/rss-{1}-{2}-{3}.json' bash '$RUN/scripts/memory_fit_worker.sh' {1} {2} {3} > '$RUN/memory/worker-{1}-{2}-{3}.log' 2>&1" :::: "$RUN/tasks.tsv"
seq 1 7 | parallel --jobs 4 --halt soon,fail=1 --joblog "$RUN/audit-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c1 --mem=2G /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project='$RUN/env' '$FROZEN_RUN/source/benchmark/selection/sparse/audit.jl' {} > '$RUN/memory/audit-{}.log' 2>&1"
python3 "$RUN/scripts/memory_fit_summary.py"
echo FIT_ONLY_MEMORY_VALIDATED
