#!/bin/bash
set -euo pipefail
seed="$1"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
if (( seed % 2 )); then order="baseline candidate"; else order="candidate baseline"; fi
for mode in $order; do
  if [ "$mode" = baseline ]; then
    env="$prior/env-opt-v4-release"
    export JULIA_DEPOT_PATH="$prior/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
  else
    env="$PWD/env-gap-release"
    export JULIA_DEPOT_PATH="$PWD/depot-gap-release:$prior/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
  fi
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$env" harness-holdout/holdout-source.jl "$seed" "$PWD/holdout" holdout/julia "$mode"
done
