#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-setup
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=setup-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BENCH_GUROBI="${BENCH_GUROBI:-1}"
if [ "$BENCH_GUROBI" = 1 ]; then
  module load gurobi/12.0.3
  export GUROBI_JL_USE_GUROBI_JLL=false
fi
J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
# Optional commit archives created during staging; source directories must
# otherwise already contain the recorded source revisions.
for f in CoordinateDescent.jl ProximalBase.jl ProximalSolvers.jl; do
  if [ -f "$f.tar" ]; then tar -xf "$f.tar"; fi
done
"$J" --project=. -e 'using Pkg; Pkg.develop([PackageSpec(path="source/ProximalBase.jl"),PackageSpec(path="source/ProximalSolvers.jl"),PackageSpec(path="source/CoordinateDescent.jl")]); Pkg.instantiate(); get(ENV,"BENCH_GUROBI","0")=="1" && Pkg.build("Gurobi"); Pkg.precompile(); Pkg.status()'
"$J" --project=. validate.jl > validation.log 2>&1
"$J" --project=. worker.jl small 101 2 pilot pilot-results > pilot.log 2>&1
