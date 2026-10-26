#!/bin/bash
set -euo pipefail
engine=$1
caseid=$2
rep=$3
stem=outputs/gaussian-native-full/$caseid/$rep.$engine
[[ -f $stem.complete ]] && exit 0
if [[ $engine == EconML ]]; then
 /home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/venv/bin/python ../gaussian-native-full-v1/gaussian-native-full.py "$engine" "$caseid" "$rep"
else
 apptainer exec --bind /home1 /apps/containers/r/4.5/r-4.5.3.sif Rscript ../gaussian-native-full-v1/gaussian-native-full.R "$engine" "$caseid" "$rep"
fi
