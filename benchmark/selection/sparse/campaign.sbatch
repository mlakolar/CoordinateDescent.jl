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
# PS335 guard suites run once in coordinated job13424116, a dependency.
# Targeted335estimator tests already passed in13424143; reuse their receipt.
# Calibrate the previously untested density strata before repeated timing.
if ! grep -q AUDIT_DONE "$RUN/calibration-density/audit.log" 2>/dev/null; then
export RESULTS_DIR=calibration-density CASES=5,6,7
mkdir -p "$RUN/$RESULTS_DIR"
for id in 5 6 7; do
 for engine in native baseline; do
  cp "$RUN/results/coef-1-$id-$engine.csv" "$RUN/$RESULTS_DIR/coef-0-$id-$engine.csv"
 done
done
for engine in SLOPE.jl R; do
 if ! grep -q 'case=7' "$RUN/$RESULTS_DIR/$engine.log" 2>/dev/null; then
  bash "$source_dir/worker.sh" "$engine" 0 timing > "$RUN/$RESULTS_DIR/$engine.log" 2>&1
 fi
done
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/$ENV_DIR" "$source_dir/audit.jl" 0 > "$RUN/$RESULTS_DIR/audit.log" 2>&1
fi
export RESULTS_DIR=results CASES=1,2,3,4,5,6,7,8,9,10,11
# Serialize timing processes; rotate backend order between replications.
python3 - <<'PY' > "$RUN/tasks.tsv"
import csv,json,os
from pathlib import Path
out=Path(os.environ['RUN'])/'results'
def complete(engine,rep):
 p=out/f'timing-{engine}-{rep}.csv';receipt=out/f'process-rss-{engine}-{rep}.json'
 if not p.exists() or not receipt.exists():return False
 with p.open() as f:rows=list(csv.DictReader(f))
 expected=30 if engine=='baseline' else 33
 return json.loads(receipt.read_text())['exit_code']==0 and len(rows)==expected
engines=['native','baseline','SLOPE.jl','R']
for rep in range(1,8):
 order=engines[(rep-1)%4:]+engines[:(rep-1)%4]
 if rep%2==0: order=order[::-1]
 for engine in order:
  if not complete(engine,rep):print(engine,rep,sep='\t')
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
