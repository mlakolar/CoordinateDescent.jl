#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:06:00
#SBATCH --job-name=cd80-gram-check
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=gram-check-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1
for version in release lts; do
  env="env-opt-v4-$version"; mkdir -p "$env"
  for f in Project.toml Manifest.toml; do
    sed "s|$old/source-fixed|$PWD/source-opt-v4|g" "$old/env-$version/$f" > "$env/$f"
  done
  export JULIA_DEPOT_PATH="$PWD/depot-opt-v4-$version:$old/depot-$version:$old/depot:/home1/mkolar/juliaPkg/depot:"
  if [ "$version" = release ]; then julia=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia; else julia=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia; fi
  "$julia" --startup-file=no --project="$env" --depwarn=error -e 'using CoordinateDescent, Test, Aqua; @testset "Cached fused paths" begin; include("source-opt-v4/test/structured.jl"); include("source-opt-v4/test/structured_golden.jl"); Aqua.test_all(CoordinateDescent;stale_deps=(ignore=[:KernelSmoothing],)); end'
done
