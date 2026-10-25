#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:35:00
set -euo pipefail
export BASE_RUN="$RUN"
export JULIA_DEPOT_PATH="$BASE_RUN/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load gcc/13.3.0 apptainer parallel/20240522
if [[ ${CANARY:-0} == 1 ]]; then
  export RUN="$BASE_RUN/${PILOT_NAME:-pilot}" MC_B=2 EXPECTED_REPS=1
  mkdir -p "$RUN/results"
  srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash "$BASE_RUN/source/benchmark/selection/compare/worker.sh" canary 1 > "$RUN/worker.log" 2>&1
  python3 "$BASE_RUN/source/benchmark/selection/compare/summarize.py"
else
  # Benchmark steps are serialized to avoid contention between this task's workers.
  # Seven processes use identical data; each case has three timed batches.
  export MC_B=20
  /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$BASE_RUN/env" "$BASE_RUN/source/benchmark/selection/compare/run.jl" generate 1 > "$RUN/generate.log" 2>&1
  seq 1 7 | parallel --jobs 1 --halt soon,fail=1 --joblog "$RUN/benchmark-joblog.tsv" \
    "srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash '$BASE_RUN/source/benchmark/selection/compare/worker.sh' benchmark {} > '$RUN/benchmark-{}.log' 2>&1"
  seq 1 7 | parallel --jobs 2 --halt soon,fail=1 --joblog "$RUN/mc-joblog.tsv" \
    "srun --exclusive --exact -N1 -n1 -c2 --mem=6G bash '$BASE_RUN/source/benchmark/selection/compare/worker.sh' mc {} > '$RUN/mc-{}.log' 2>&1"
  python3 "$BASE_RUN/source/benchmark/selection/compare/summarize.py"
fi
