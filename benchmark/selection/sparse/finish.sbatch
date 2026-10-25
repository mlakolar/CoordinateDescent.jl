#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=epyc-7313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ENV_DIR=env335
module load gcc/13.3.0 apptainer parallel/20240522
source_dir="$RUN/source/benchmark/selection/sparse"
if [[ -f "$RUN/results/validation.json" && -f "$RUN/results/profile-summary.csv" ]]; then
 echo ALREADY_VALIDATED
 exit 0
fi
# Resume only missing or failed fresh-process memory records, never timed fits.
python3 - <<'PY' > "$RUN/memory-remaining.tsv"
import json,os
from pathlib import Path
root=Path(os.environ['RUN'])/'memory'
for rep in range(1,8):
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if case==11 and engine=='baseline':continue
   p=root/f'rss-{engine}-{rep}-{case}.json'
   if p.exists() and json.loads(p.read_text()).get('exit_code')==0:continue
   print(engine,rep,case,sep='\t')
PY
export RESULTS_DIR=memory
parallel --colsep '\t' --jobs 1 --halt soon,fail=1 --joblog "$RUN/memory-continuation-joblog.tsv" \
 "srun --exclusive --exact -N1 -n1 -c2 --mem=4G python3 '$source_dir/measure.py' '$RUN/memory/rss-{1}-{2}-{3}.json' bash '$source_dir/worker.sh' {1} {2} memory {3} > '$RUN/memory/worker-{1}-{2}-{3}.log' 2>&1" :::: "$RUN/memory-remaining.tsv"
for rep in {1..7}; do
 if [[ ! -f "$RUN/results/accuracy-$rep.csv" || ! -f "$RUN/memory/accuracy-$rep.csv" ]] || ! grep -q AUDIT_DONE "$RUN/results/audit-$rep.log"; then
  bash "$source_dir/audit_worker.sh" "$rep" > "$RUN/results/audit-$rep.log" 2>&1
 fi
done
python3 "$source_dir/summarize.py"
python3 "$source_dir/profile_summary.py"
echo FINISH_DONE
