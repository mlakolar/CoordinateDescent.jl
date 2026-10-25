#!/bin/bash
set -euo pipefail
seed="$1"
root="$PWD"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-stopping
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_DEPOT_PATH="$root/depot-release:$prior/depot-gap-v2-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
input="$root/inputs-$seed"; refs="$root/reference-$seed"
apptainer exec --cleanenv --env R_LIBS_USER="$old/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,TMPDIR="$old/tmp" \
 /apps/containers/r/4.5/r-4.5.3.sif Rscript "$root/harness/generate.R" "$input" "$seed"
apptainer exec --cleanenv --env R_LIBS_USER="$old/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,TMPDIR="$old/tmp" \
 /apps/containers/r/4.5/r-4.5.3.sif Rscript "$root/harness/reference.R" "$seed" "$input" "$refs"
if [ "$seed" = 8502 ]; then order="candidate baseline"; else order="baseline candidate"; fi
for setting in $order; do
 if [ "$setting" = baseline ]; then env="$root/env-baseline-release"; else env="$root/HighDimensionalModels.jl/.master-test-env/v1.13"; fi
 /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$env" \
  "$root/harness/measure.jl" "$seed" "$input" "$refs" "$root/holdout" "$setting"
done
