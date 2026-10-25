#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd80-opt-check
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=opt-check-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
for version in release lts; do
  env="env-opt-v3-$version"; mkdir -p "$env"
  for f in Project.toml Manifest.toml; do
    sed "s|$old/source-fixed|$PWD/source-opt-v3|g" "$old/env-$version/$f" > "$env/$f"
  done
  export JULIA_DEPOT_PATH="$old/depot-$version:$old/depot:/home1/mkolar/juliaPkg/depot:"
  if [ "$version" = release ]; then julia=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia; else julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia; fi
  "$julia" --startup-file=no --project="$env" --depwarn=error -e 'using CoordinateDescent, Test, Aqua; @testset "Optimized structured fits" begin; include("source-opt-v3/test/structured.jl"); include("source-opt-v3/test/structured_golden.jl"); Aqua.test_all(CoordinateDescent;stale_deps=(ignore=[:KernelSmoothing],)); end'
done
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release"
export JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
mkdir -p optimized-canary-v3
for seed in 8201 8202 8203; do
  bash harness/worker.sh "$seed" Julia canary optimized-canary-v3
done
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness/check.jl inputs optimized-canary-v3 baseline-canary-v2 baseline-full optimized-canary-v3
