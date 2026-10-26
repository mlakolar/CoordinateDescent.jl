#!/bin/bash
set -u
case=$1;out="results/pilot-cached-recovery-v3/$case";mkdir -p "$out"
python3 code/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 300s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-baseline-1.13 code/recover_cached.jl "$case" "$out" > "$out/worker.log" 2>&1
