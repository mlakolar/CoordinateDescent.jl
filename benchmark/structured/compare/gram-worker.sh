#!/bin/bash
set -euo pipefail
seed="$1"
export JULIA_PROJECT_ENV="$PWD/env-opt-v4-release"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export COMPARE_DEPOT_PATH="$PWD/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
export COMPARE_TIGHT=true
bash harness-gram/worker.sh "$seed" Julia gram optimized-gram-matched
export COMPARE_TIGHT=false
bash harness-gram/worker.sh "$seed" Julia gram optimized-gram-default
export COMPARE_TIGHT=true
bash harness-gram/worker.sh "$seed" Julia gram-profile optimized-gram-profile
