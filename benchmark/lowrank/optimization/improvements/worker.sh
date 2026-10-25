#!/bin/bash
set -euo pipefail
case_id=$1
language=$2
root=$PWD
mkdir -p "results/main/$case_id/$language"
case "$language" in
 julia) "$JULIA" --startup-file=no --project=env-1.13.2 code/resume_worker.jl "inputs/$case_id" "results/main/$case_id/julia" after ;;
 before) "$JULIA" --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/env code/resume_worker.jl "inputs/$case_id" "results/before/$case_id/julia" before ;;
 python) "$PYTHON" code/python_worker.py "inputs/$case_id" "results/main/$case_id/python" --phase main ;;
 r) Rscript code/r_worker.R "inputs/$case_id" "results/main/$case_id/r" main ;;
 r-component) Rscript code/r_worker.R "inputs/$case_id" "results/main/$case_id/r-component" main component_only ;;
 *) exit 2 ;;
esac
