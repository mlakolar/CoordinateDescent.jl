#!/bin/bash
set -euo pipefail
rep=$1
src="$RUN/source/benchmark/selection/compare"
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$RUN/env" "$src/gaussian.jl" "$rep"
# run.R's helper preamble reads commandArgs[2] as rep; pass a dummy mode first.
apptainer exec --cleanenv --env "RUN=$RUN,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" \
  /apps/containers/r/4.5/r-4.5.3.sif Rscript "$src/gaussian.R" helpers "$rep"
