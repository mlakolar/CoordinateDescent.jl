#!/bin/bash
set -euo pipefail
stage=$1
shift
case "$stage" in
generate) /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$HDM_BASE/env-comparison-1.10" "$HDM_CODE/runtime_generate.jl" "$@" ;;
native|julia) /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$HDM_BASE/env-comparison-1.10" "$HDM_CODE/runtime_fit.jl" "$stage" "$@" ;;
R) apptainer exec --cleanenv --env "HDM_STUDY=$HDM_STUDY,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" /apps/containers/r/4.5/r-4.5.3.sif Rscript "$HDM_CODE/runtime_fit.R" "$@" ;;
check) /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_CODE/runtime_check.py" "$@" ;;
*) echo "unknown stage $stage" >&2;exit 2 ;;
esac
