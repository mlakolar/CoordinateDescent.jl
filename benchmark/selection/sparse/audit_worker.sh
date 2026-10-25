#!/bin/bash
set -euo pipefail
julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
script="$RUN/source/benchmark/selection/sparse/audit.jl"
RESULTS_DIR=results CASES=1,2,3,4,5,6,7,8,9,10,11 "$julia" --startup-file=no --project="$RUN/env" "$script" "$1"
RESULTS_DIR=memory CASES=4,10,11 "$julia" --startup-file=no --project="$RUN/env" "$script" "$1"
