#!/bin/bash
set -euo pipefail
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
if [ "$1" = warmfix ]; then
  export JULIA_DEPOT_PATH="$prior/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
  for seed in 8201 8202 8203; do
    /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$prior/env-opt-v4-release" harness-validation/calibrate-warmfix.jl "$seed" "$prior" calibration-warmfix baseline,gap1e6,gap3e7,gap1e7
  done
else
  version="$1"
  export JULIA_DEPOT_PATH="$PWD/depot-gap-$version:$prior/depot-opt-v4-$version:$old/depot-$version:$old/depot:/home1/mkolar/juliaPkg/depot:"
  export CD_STRUCTURED_GOLDEN="$old/source-fixed/test/golden/structured"
  if [ "$version" = release ]; then julia=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia; else julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia; fi
  "$julia" --startup-file=no --project="env-gap-$version" --depwarn=error -e 'using CoordinateDescent,Test,Aqua; @testset "Linear signal stopping" begin; include("source-gap-v1/test/structured.jl");include("source-gap-v1/test/structured_golden.jl");Aqua.test_all(CoordinateDescent;stale_deps=(ignore=[:KernelSmoothing],));end'
fi
