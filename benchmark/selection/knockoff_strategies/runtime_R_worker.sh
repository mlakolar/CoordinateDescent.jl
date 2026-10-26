#!/bin/bash
set -euo pipefail
case=$1
rep=$2
python3 "$HDM_KNOCKOFF_CODE/resource_wrapper.py" "$HDM_KNOCKOFF_RUNTIME/resource-R-$case-$rep.json" apptainer exec --cleanenv --env "HDM_KNOCKOFF_RUNTIME=$HDM_KNOCKOFF_RUNTIME,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" /apps/containers/r/4.5/r-4.5.3.sif Rscript "$HDM_KNOCKOFF_CODE/runtime_R.R" "$case" "$rep"
