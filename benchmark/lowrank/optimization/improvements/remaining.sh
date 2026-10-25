#!/bin/bash
# Downstream stages from comparison.sbatch, with corrected Python CLI aliases.
"$CD81_PARALLEL_BIN" -j2 --joblog results/recovery/r-component-joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/worker-v3.sh responses-{} r-component > results/main/responses-{}-r-component.log 2>&1' ::: 1 2 3 4 5
"$CD81_PARALLEL_BIN" -j8 --joblog results/adaptive/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env-1.13.2 code/adaptive.jl {} > results/adaptive/rep-{}.log 2>&1' ::: 1 2 3 4 5
CD81_CORE_ONLY=1 srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/analyze-v8.py
"$PYTHON" code/collect-v8.py
for solver in before_rank cd_rank cd_nuclear cd_panoc before_nuclear cd_sparse_api cd_sparse_fista before_sparse cd_sparse block_fista ps_fista ps_admm pa_fista pa_panoc cd_component component_fista scipy_rrr py_fista py_admm r_rrr r_rrr_qr r_srrr r_component;do
 case "$solver" in before_*) language=before;;scipy_rrr|py_*) language=python;;r_*) language=r;;*) language=julia;;esac
 for name in responses wide;do
  [ "$solver" = r_component ] && [ "$name" = responses ] && continue
  for rep in 1 2 3 4 5;do printf '%s-%s %s %s\n' "$name" "$rep" "$language" "$solver";done
 done
done > memory-tasks.txt
"$CD81_PARALLEL_BIN" -j8 --colsep ' ' --joblog results/memory/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/memory-v3.sh {1} {2} {3} > results/memory/{1}-{3}.log 2>&1' :::: memory-tasks.txt
"$CD81_PARALLEL_BIN" -j2 --joblog results/recovery/r-memory-joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/memory-v3.sh responses-{} r r_component > results/memory/responses-{}-r_component.log 2>&1' ::: 1 2 3 4 5
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/memory_summary.py
"$PYTHON" code/collect-v8.py
