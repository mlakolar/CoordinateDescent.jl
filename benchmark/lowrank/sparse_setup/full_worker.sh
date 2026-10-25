#!/bin/bash
set -u
label=$1;phase=$2;root=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight
rep=${label##*-};status=0
harness=${HDM_HARNESS:-full-v2}
variants=(public-baseline public-candidate common-baseline common-candidate r-public r-common)
for ((i=0;i<6;i++));do
  key=${variants[$(((i+rep)%6))]};out="results/$phase/$label/$key";mkdir -p "$out"
  if [[ $key == r-* ]];then
    mode=${key#r-}
    python3 "$harness"/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 900s bash "$harness"/r_runner.sh "inputs/$label" "$out" "$mode" > "$out/worker.log" 2>&1
  else
    mode=${key%-*};variant=${key##*-}
    if [[ $variant == baseline ]];then envdir="$root/issue99/env-baseline-1.13";else envdir="$root/issue102/env-candidate2-1.13";fi
    python3 "$harness"/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 900s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" "$harness"/full_worker.jl "inputs/$label" "$out" "$variant" "$mode" > "$out/worker.log" 2>&1
  fi
  code=$?;printf '%s\t%s\n' "$key" "$code" >> "results/$phase/$label/outcomes.tsv"
  ((code==0)) || status=1
done
pathvariants=(baseline candidate)
for ((i=0;i<2;i++));do
  variant=${pathvariants[$(((i+rep)%2))]}
  if [[ $variant == baseline ]];then envdir="$root/issue99/env-baseline-1.13";else envdir="$root/issue102/env-candidate2-1.13";fi
  out="results/$phase/$label/path-$variant";mkdir -p "$out"
  python3 "$harness/resource_wrapper.py" "$out/resource.json" timeout --kill-after=15s 900s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" "$harness/path_components.jl" "inputs/$label" "$out" "$variant" "results/$phase/$label/public-$variant" > "$out/worker.log" 2>&1
  code=$?;printf '%s\t%s\n' "path-$variant" "$code" >> "results/$phase/$label/outcomes.tsv"
  ((code==0)) || status=1
done
exit "$status"
