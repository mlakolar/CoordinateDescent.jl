#!/bin/bash
set -u
label=$1
phase=$2
rep=${label##*-}
variants=(baseline candidate python r)
if [[ $# -ge 3 ]]; then variants=("$3"); fi
offset=$((rep % 4))
status=0
for ((k=0;k<${#variants[@]};k++)); do
  variant=${variants[$(((k+offset)%${#variants[@]}))]}
  dest="results/$phase/$label/$variant"
  mkdir -p "$dest"
  case "$variant" in
    baseline) command=("$JULIA" --startup-file=no --project="$BASE_ENV" julia_worker.jl "inputs/$label" "$dest" baseline);;
    candidate) command=("$JULIA" --startup-file=no --project="$CANDIDATE_ENV" julia_worker.jl "inputs/$label" "$dest" candidate);;
    python) command=("$PYTHON" python_worker.py "inputs/$label" "$dest");;
    r) command=(Rscript r_worker.R "inputs/$label" "$dest");;
  esac
  python3 resource_wrapper.py "$dest/resource.json" timeout --signal=TERM --kill-after=10s 240s "${command[@]}" > "$dest/worker.log" 2>&1
  outcome=$?
  printf '%s\t%s\n' "$variant" "$outcome" >> "results/$phase/$label/outcomes.tsv"
  ((outcome==0)) || status=1
done
exit "$status"
