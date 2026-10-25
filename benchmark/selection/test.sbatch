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
source benchmark/selection/cache_inputs.sh
export JULIA_DEPOT_PATH="$RUN/depot:$RUN/shared-inputs"
export CD_SELECTION_GOLDEN="$RUN/golden"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'using Pkg; Pkg.test("HighDimensionalModels";julia_args=["--depwarn=error"])' > "$RUN/full-tests.log" 2>&1
