#!/bin/bash
set -u
case=$1;config=$2;out="results/pilot-public-recovery-v2/$case/$config";mkdir -p "$out"
python3 code/resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 180s /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-baseline-1.13 code/recover_public.jl "$case" "$config" "$out" > "$out/worker.log" 2>&1
