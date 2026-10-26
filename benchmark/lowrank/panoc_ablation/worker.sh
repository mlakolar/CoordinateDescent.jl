#!/bin/bash
set -u
case=$1;out="results/pilot-v1/$case";mkdir -p "$out"
python3 code/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 1800s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-baseline-1.13 code/worker.jl "$case" "$out" > "$out/worker.log" 2>&1
