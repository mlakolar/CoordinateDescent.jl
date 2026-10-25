#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$RUN/source"
source benchmark/selection/cache_inputs.sh
export JULIA_DEPOT_PATH="$RUN/depot:$RUN/shared-inputs"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CD_SELECTION_GOLDEN="$RUN/golden"
# Reuse the exact dependency sources already resolved by the 1.13 environment.
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'using Pkg, TOML; d=Dict(p.name=>p.source for p in values(Pkg.dependencies()) if p.name in ("ProximalBase","ProximalSolvers")); open(ENV["RUN"]*"/dependency-paths.toml","w") do io; TOML.print(io,d); end'
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no -e 'using Pkg,TOML; d=TOML.parsefile(ENV["RUN"]*"/dependency-paths.toml"); Pkg.activate(ENV["RUN"]*"/env110"); Pkg.develop([PackageSpec(path=d["ProximalBase"]),PackageSpec(path=d["ProximalSolvers"]),PackageSpec(path=pwd())]); Pkg.add("DelimitedFiles"); using CoordinateDescent,Test; include("test/selection.jl"); include("test/selection_golden.jl")' > "$RUN/compatibility-110.log" 2>&1
