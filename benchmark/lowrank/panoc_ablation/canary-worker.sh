#!/bin/bash
set -euo pipefail
v=$1;version=1.13.2;[[ $v != 1.10 ]] || version=1.10.12
/home1/mkolar/juliaPkg/julia-$version/bin/julia --startup-file=no --project=env-candidate1-$v code/canary.jl > results/prototype-canary-v1/tests-$v.log 2>&1
