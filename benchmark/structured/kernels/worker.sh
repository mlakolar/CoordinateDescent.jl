#!/bin/bash
set -u
label=$1; phase=$2
root=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight
variants=(baseline candidate)
rep=${label##*-}
if ((rep%2)); then variants=(candidate baseline); fi
status=0
for variant in "${variants[@]}"; do
  if [[ $variant == baseline ]]; then envdir="$root/issue99/env-baseline-1.13"; else envdir="$root/issue98/env-candidate-1.13"; fi
  out="results/$phase/$label/$variant";mkdir -p "$out"
  python3 resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 1800s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" julia_worker.jl "inputs/$label" "$out" "$variant" > "$out/worker.log" 2>&1
  code=$?;printf '%s\t%s\n' "$variant" "$code" >> "results/$phase/$label/outcomes.tsv"
  ((code==0)) || status=1
done
exit "$status"
