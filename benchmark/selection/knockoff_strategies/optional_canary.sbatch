#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=hdm106-optional-canary
#SBATCH --comment=20261026-hdm106-optional-canary-v3
#SBATCH --output=optional-v3-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
mkdir -p optional-v3-results
for version in 1.10 1.13; do
 binary=/home1/mkolar/juliaPkg/julia-$([[ $version == 1.10 ]] && echo 1.10.12 || echo 1.13.2)/bin/julia
 srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$binary" --startup-file=no optional-v3-code/optional_canary.jl "$version" > "optional-v3-results/optional-$version.log" 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
exit "$status"
