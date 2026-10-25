#!/bin/bash
set -u
label=$1;phase=$2;harness=${HDM_HARNESS:-full-v2};status=0
root=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight
mkdir -p "results/$phase/$label"
for key in public-baseline public-candidate common-baseline common-candidate;do
 ln -s "$PWD/results/pilot-full-v1/$label/$key" "results/$phase/$label/$key"
done
for mode in public common;do
 out="results/$phase/$label/r-$mode";mkdir -p "$out"
 python3 "$harness/resource_wrapper.py" "$out/resource.json" timeout --kill-after=15s 900s bash "$harness/r_runner.sh" "inputs/$label" "$out" "$mode" > "$out/worker.log" 2>&1 || status=1
done
for variant in baseline candidate;do
 if [[ $variant == baseline ]];then envdir="$root/issue99/env-baseline-1.13";else envdir="$root/issue102/env-candidate2-1.13";fi
 out="results/$phase/$label/path-$variant";mkdir -p "$out"
 python3 "$harness/resource_wrapper.py" "$out/resource.json" timeout --kill-after=15s 900s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" "$harness/path_components.jl" "inputs/$label" "$out" "$variant" "results/$phase/$label/public-$variant" > "$out/worker.log" 2>&1 || status=1
done
exit "$status"
