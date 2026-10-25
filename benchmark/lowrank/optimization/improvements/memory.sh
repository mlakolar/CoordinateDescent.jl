#!/bin/bash
set -euo pipefail
case_id=$1
language=$2
solver=$3
out="results/memory/$case_id/$solver"
mkdir -p "$out"
case "$language" in
 julia) "$JULIA" --startup-file=no --project=env-1.13.2 code/updated_worker.jl "inputs/$case_id" "$out" after "$solver" ;;
 before) "$JULIA" --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/env code/updated_worker.jl "inputs/$case_id" "$out" before "$solver" ;;
 python) "$PYTHON" code/python_worker.py "inputs/$case_id" "$out" --memory "$solver" ;;
 r) Rscript code/r_worker.R "inputs/$case_id" "$out" main memory "$solver" ;;
 *) exit 2 ;;
esac
