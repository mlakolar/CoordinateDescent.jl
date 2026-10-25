#!/bin/bash
set -euo pipefail
case_id=$1
language=$2
solver=$3
dest="results/memory/$case_id/$solver"
mkdir -p "$dest"
case "$language" in
  julia) "$JULIA" --startup-file=no --project=env code/julia_worker.jl "inputs/$case_id" "$dest" main "$solver" ;;
  python) "$PYTHON" code/python_worker.py "inputs/$case_id" "$dest" --memory "$solver" ;;
  r) Rscript code/r_worker.R "inputs/$case_id" "$dest" main memory "$solver" ;;
esac
