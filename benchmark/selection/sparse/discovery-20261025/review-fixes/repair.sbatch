#!/bin/bash
#SBATCH --job-name=cd82-repair-74bc14b
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:11:00
#SBATCH --output=/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/repair-74bc14b/slurm-%j.log
set -euo pipefail
export FIX_ROOT=/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/repair-74bc14b
export JULIA_DEPOT_PATH="$FIX_ROOT/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/renamed/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 parallel/20240522
srun --exclusive --exact -N1 -n1 -c2 --mem=6G /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no "$FIX_ROOT/setup.jl" > "$FIX_ROOT/setup-110.log" 2>&1
seq 1 3 | parallel --jobs 2 --halt soon,fail=1 --joblog "$FIX_ROOT/regression-joblog.tsv" \
  "srun --exclusive --exact -N1 -n1 -c2 --mem=6G timeout --signal=TERM --kill-after=5s 120s /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --depwarn=error --project='$FIX_ROOT/env-1.10' '$FIX_ROOT/regressions.jl' {} > '$FIX_ROOT/regression-{}.log' 2>&1"
srun --exclusive --exact -N1 -n1 -c2 --mem=6G /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no "$FIX_ROOT/setup.jl" > "$FIX_ROOT/setup-113.log" 2>&1
srun --exclusive --exact -N1 -n1 -c2 --mem=6G /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --check-bounds=yes --depwarn=error --project="$FIX_ROOT/env-1.13" "$FIX_ROOT/HighDimensionalModels.jl/test/runtests.jl" > "$FIX_ROOT/full-hdm-113.log" 2>&1 &
hdm_pid=$!
srun --exclusive --exact -N1 -n1 -c2 --mem=6G /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --check-bounds=yes --depwarn=yes --project="$FIX_ROOT/env-1.13" "$FIX_ROOT/ProximalOPT.jl/test/runtests.jl" > "$FIX_ROOT/full-ps-113.log" 2>&1 &
ps_pid=$!
hdm_status=0; wait "$hdm_pid" || hdm_status=$?
ps_status=0; wait "$ps_pid" || ps_status=$?
printf 'HDM_FULL_STATUS=%s\nPS_FULL_STATUS=%s\n' "$hdm_status" "$ps_status"
test "$hdm_status" -eq 0 && test "$ps_status" -eq 0
echo REPAIR_VALIDATED
