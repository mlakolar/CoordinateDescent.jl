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
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Rendering the docs runs their Julia examples, so it also stays on Discovery.
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate(); s=read("docs/make.jl",String); s=replace(s, "makedocs(" => "makedocs(remotes=nothing,"); s=replace(s, r"deploydocs\([^\n]+\)" => "nothing"); include_string(Main,s,joinpath(pwd(),"docs","make.jl"))' > "$RUN/docs.log" 2>&1
