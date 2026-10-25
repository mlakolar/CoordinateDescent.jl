#!/bin/bash
set -euo pipefail
version="$1"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-stopping
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_DEPOT_PATH="$PWD/depot-$version:$prior/depot-gap-v2-$version:$old/depot-$version:$old/depot:/home1/mkolar/juliaPkg/depot:"
if [ "$version" = release ]; then julia=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia; minor=1.13; else julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia; minor=1.10; fi
"$julia" --startup-file=no --project="HighDimensionalModels.jl/.master-test-env/v$minor" --depwarn=error -e 'using HighDimensionalModels,Test,Aqua; @testset "Final issue97 accuracy checks" begin;include("harness-final/structured-final.jl");include("HighDimensionalModels.jl/test/structured_golden.jl");Aqua.test_all(HighDimensionalModels;stale_deps=(ignore=[:KernelSmoothing],));end'
