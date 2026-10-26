#!/bin/bash
set -euo pipefail
rep=$1
variants=(baseline candidate)
(( rep % 2 == 0 )) && variants=(candidate baseline)
for variant in "${variants[@]}"; do
  python3 resource_wrapper.py "results/resource-$variant-$rep.json" /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="env-$variant-1.13" bench.jl "$variant" "$rep" results > "results/worker-$variant-$rep.log" 2>&1
done
