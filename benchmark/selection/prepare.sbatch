#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:35:00
set -euo pipefail
cd "$RUN/source"
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SELECTION_RLIB="$RUN/Rlib"
module load apptainer
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no benchmark/selection/setup.jl > "$RUN/setup.log" 2>&1
srun --exclusive --exact -N1 -n1 -c2 /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'using HighDimensionalModels, Test; include("test/selection.jl")' > "$RUN/unit.log" 2>&1 &
unit_pid=$!
apptainer exec --cleanenv --env "OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,SELECTION_RLIB=$RUN/Rlib,R_LIBS_USER=$RUN/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" /apps/containers/r/4.5/r-4.5.3.sif Rscript benchmark/selection/setup.R > "$RUN/R-setup.log" 2>&1 &
r_pid=$!
status=0
wait "$unit_pid" || status=1
wait "$r_pid" || status=1
exit "$status"
