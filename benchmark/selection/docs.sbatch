#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$RUN/source"
source benchmark/selection/cache_inputs.sh
export JULIA_DEPOT_PATH="$RUN/depot:$RUN/shared-inputs"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Verify the shipped fixture path without the run-specific environment override.
unset CD_SELECTION_GOLDEN
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'include("test/selection_golden.jl")' > "$RUN/shipped-fixtures.log" 2>&1
# Rendering the docs runs their Julia examples, so it also stays on Discovery.
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate(); s=read("docs/make.jl",String); s=replace(s, "makedocs(" => "makedocs(root=joinpath(pwd(),\"docs\"),remotes=nothing,"); s=replace(s, r"deploydocs\([^\n]+\)" => "nothing"); include_string(Main,s,joinpath(pwd(),"docs","make.jl"))' > "$RUN/docs.log" 2>&1
