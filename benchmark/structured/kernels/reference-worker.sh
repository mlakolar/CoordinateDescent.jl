#!/bin/bash
set -euo pipefail
module load apptainer/1.3.6
label=$1;phase=$2;gridphase=${3:-main}
out="results/$phase/$label/reference";mkdir -p "$out" tmp
python3 resource_wrapper.py "$out/resource.json" timeout --kill-after=15s 2400s apptainer exec --cleanenv \
  --env "R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,TMPDIR=$PWD/tmp" \
  /apps/containers/r/4.5/r-4.5.3.sif Rscript reference.R "$label" "$PWD/inputs" "$PWD/$out" "$gridphase" > "$out/worker.log" 2>&1
