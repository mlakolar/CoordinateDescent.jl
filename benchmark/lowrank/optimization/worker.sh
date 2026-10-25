#!/bin/bash
set -euo pipefail
case_id=$1
language=$2
phase=$3
dest="results/$phase/$case_id/$language"
mkdir -p "$dest"
case "$language" in
  julia) "$JULIA" --startup-file=no --project=env code/julia_worker.jl "inputs/$case_id" "$dest" "$phase" ;;
  python) "$PYTHON" code/python_worker.py "inputs/$case_id" "$dest" --phase "$phase" ;;
  r) Rscript code/r_worker.R "inputs/$case_id" "$dest" "$phase" ;;
  r-component) Rscript code/r_worker.R "inputs/$case_id" "$dest" "$phase" component_only ;;
esac
