#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:25:00
set -euo pipefail
cd "$RUN/source"
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'using Pkg; Pkg.test("CoordinateDescent";julia_args=["--depwarn=error"])' > "$RUN/full-tests.log" 2>&1
