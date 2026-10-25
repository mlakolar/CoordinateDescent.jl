#!/bin/bash
# Run from an immutable task directory containing source/, base/, solvers/,
# kernels/, and a task-local depot layered over the existing shared cache.
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=cd80-evaluate
#SBATCH --comment=20261025-cd80-structured
#SBATCH --output=evaluate-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
module load parallel/20240522 apptainer
"$JULIA" --startup-file=no --project=source/benchmark/structured -e 'using Pkg; Pkg.develop([PackageSpec(path=abspath("base")), PackageSpec(path=abspath("solvers")), PackageSpec(path=abspath("kernels")), PackageSpec(path=abspath("source"))]); Pkg.instantiate(); Pkg.status()'
apptainer exec --cleanenv --env R_LIBS_USER="$PWD/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3",OPENBLAS_NUM_THREADS=1,TMPDIR="$PWD/tmp" \
 /apps/containers/r/4.5/r-4.5.3.sif Rscript source/benchmark/structured/golden.R golden > golden-R.log 2>&1
"$JULIA" --startup-file=no --project=source/benchmark/structured source/benchmark/structured/check_golden.jl golden > golden-Julia.log 2>&1
mkdir -p results
parallel --jobs 3 --halt soon,fail=1 --joblog workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M "$JULIA" --startup-file=no --project=source/benchmark/structured source/benchmark/structured/run.jl {1} results > worker-{1}.log 2>&1' ::: 8101 8102 8103
