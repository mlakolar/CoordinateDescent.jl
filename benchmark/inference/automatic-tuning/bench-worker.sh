#!/bin/bash
set -euo pipefail
engine=$1
label=$2
rep=$3
prior=/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
script=source-v3/benchmark/inference/automatic-tuning
case "$engine" in
 ours|oursqp|oursqp10) cmd=(/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-v3 "$script/bench.jl") ;;
 econml) cmd=("$prior/venv/bin/python" "$script/bench.py") ;;
 *) cmd=(apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env TMPDIR="$PWD/tmp" /apps/containers/r/4.5/r-4.5.3.sif Rscript "$script/bench.R") ;;
esac
python3 "$script/measure.py" "bench/$engine-$label-$rep.time" "${cmd[@]}" "$@" > "logs/$engine-$label-$rep.log" 2>&1
