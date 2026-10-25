#!/bin/bash
set -u
label=$1;root=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight
status=0
for variant in baseline candidate; do
  if [[ $variant == baseline ]];then envdir="$root/issue99/env-baseline-1.13";else envdir="$root/issue102/env-candidate-1.13";fi
  out="results/pilot/$label/$variant";mkdir -p "$out"
  python3 resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 600s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" julia_worker.jl "inputs/$label" "$out" "$variant" > "$out/worker.log" 2>&1
  code=$?;printf '%s\t%s\n' "$variant" "$code" >> "results/pilot/$label/outcomes.tsv"
  ((code==0)) || status=1
done
exit "$status"
