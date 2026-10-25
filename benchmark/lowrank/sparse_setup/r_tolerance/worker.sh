#!/bin/bash
set -u
label=$1;phase=main-r-tol-v1;status=0
mkdir -p "results/$phase/$label"
for key in public-baseline public-candidate common-baseline common-candidate path-baseline path-candidate;do
 ln -s "$PWD/results/main/$label/$key" "results/$phase/$label/$key"
done
modes=(public common);rep=${label##*-}
for ((i=0;i<2;i++));do
 mode=${modes[$(((i+rep)%2))]};out="results/$phase/$label/r-$mode";mkdir -p "$out"
 python3 r-tol-v1/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 900s bash r-tol-v1/r_runner.sh "inputs/$label" "$out" "$mode" > "$out/worker.log" 2>&1 || status=1
done
exit "$status"
