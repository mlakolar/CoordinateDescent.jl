#!/bin/bash
set -euo pipefail
case=$1
rep=$2
if (( (case + rep) % 2 == 0 ));then
 order=(candidate baseline)
else
 order=(baseline candidate)
fi
for variant in "${order[@]}";do
 bash "$HDM_KNOCKOFF_CODE/runtime_worker.sh" "$variant" "$case" "$rep"
done
