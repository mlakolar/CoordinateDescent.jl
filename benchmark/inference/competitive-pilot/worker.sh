#!/bin/bash
set -euo pipefail
engine=$1
mode=$2
scenario=$3
rep=$4
tag="$engine-$mode-$scenario-$rep"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JULIA_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
prior=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference
export JULIA_DEPOT_PATH="$prior/depot:/home1/mkolar/juliaPkg/depot:"
if [[ -f "outputs/$tag.meta" || -f "outputs/$tag.json" ]]; then
  echo "EXISTING_RESULT $tag"
  exit 0
fi
case "$engine" in
  ours|oursqp) cmd=(/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$prior/env-v4" worker.jl) ;;
  econml) cmd=(venv/bin/python worker.py) ;;
  *) cmd=(apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env TMPDIR="$PWD/tmp" /apps/containers/r/4.5/r-4.5.3.sif Rscript worker.R) ;;
esac
set +e
python3 measure.py "outputs/$tag.time" "${cmd[@]}" "$@" > "logs/$tag.log" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  printf 'status=process_failure\nprocess_exit=%s\n' "$rc" > "outputs/$tag.failure"
fi
# Preserve per-fit failure, continue collecting the pilot rather than discarding
# scientific failures or repeatedly rerunning a deterministic error.
echo "TASK_COMPLETE $tag exit=$rc"
exit 0
