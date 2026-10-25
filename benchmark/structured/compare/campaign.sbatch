#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=cd80-compare
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=campaign-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
mode="$1"; output="$2"; engines="${3:-R Julia}"
mkdir -p "$output"
export JULIA_PROJECT_ENV="${JULIA_PROJECT_ENV:-/home1/mkolar/juliaPkg/runs/20261025-cd80-structured/env-release}"
if [ ! -f inputs/cases.tsv ]; then
  old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
  apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness/generate.R inputs
fi
parallel --jobs 3 --halt soon,fail=1 --joblog "$output/worker-jobs.tsv" \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness/worker.sh {1} {2} '"$mode $output"' > '"$output"'/worker-{1}-{2}.log 2>&1' \
  ::: 8201 8202 8203 ::: $engines
if [ "$mode" != profile ]; then
  old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
  export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl inputs "$output" baseline-canary-v2 "$output"
fi
