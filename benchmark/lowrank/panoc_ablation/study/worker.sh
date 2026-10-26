#!/bin/bash
set -u
case=$1;rep=$2;method=$3
run=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue103
out="results/study-v1/$case/rep$rep/$method";folder="inputs/study-v1/$case/rep$rep"
[[ $rep != 0 ]] || { out="results/study-canary-v1/$case/$method";folder="inputs/pilot-v1/$case"; }
mkdir -p "$out"
[[ ! -f "$out/resource.json" ]] || { echo "refuse receipt overwrite $out";exit 99; }
PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
status=0
if [[ $method == py_* ]];then
 python3 code/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 2400s "$PYTHON" study-v1-code/python_worker.py "$folder" "$method" "$out" > "$out/worker.log" 2>&1 || status=$?
else
 envdir="$run/env-baseline-1.13";[[ $method != candidate ]] || envdir="$run/env-candidate2-1.13"
 python3 code/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 2400s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$envdir" study-v1-code/worker.jl "$folder" "$method" "$out" > "$out/worker.log" 2>&1 || status=$?
fi
if [[ -f "$out/fits.toml" || -f "$out/fits.json" ]];then
 "$PYTHON" study-v1-code/validate.py "$folder" "$out" > "$out/validation.log" 2>&1 || status=2
fi
exit "$status"
