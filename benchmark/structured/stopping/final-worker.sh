#!/bin/bash
set -euo pipefail
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
if [ "$1" = release ] || [ "$1" = lts ]; then
  version="$1"
  export JULIA_DEPOT_PATH="$PWD/depot-gap-v2-$version:$PWD/depot-gap-$version:$prior/depot-opt-v4-$version:$old/depot-$version:$old/depot:/home1/mkolar/juliaPkg/depot:"
  export CD_STRUCTURED_GOLDEN="$old/source-fixed/test/golden/structured"
  if [ "$version" = release ]; then julia=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia; else julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia; fi
  "$julia" --startup-file=no --project="env-gap-v2-$version" --depwarn=error -e 'using CoordinateDescent,Test,Aqua; @testset "Final linear stopping" begin; include("source-gap-v2/test/structured.jl");include("source-gap-v2/test/structured_golden.jl");Aqua.test_all(CoordinateDescent;stale_deps=(ignore=[:KernelSmoothing],));end'
else
  export JULIA_DEPOT_PATH="$PWD/depot-gap-v2-release:$PWD/depot-gap-release:$prior/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-gap-v2-release harness-final/holdout-source-final.jl "$1" "$PWD/holdout" holdout/julia-final candidate
fi
