#!/bin/bash
set -euo pipefail
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261025-cd77-bias-tuning/env-v5 ../source-v2/pilot.jl "$@"
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env TMPDIR="$PWD/tmp" /apps/containers/r/4.5/r-4.5.3.sif Rscript ../source-v2/pilot.R "$@"
