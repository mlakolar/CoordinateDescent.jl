#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd81-solvers-prep-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=prep-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_PKG_PRECOMPILE_AUTO=0
unset PARALLEL
mkdir -p results/prep env-1.13.2 env-1.10.12
cp /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/env/{Project,Manifest}.toml env-1.13.2/
cp /home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/env-final-1.10.12/{Project,Manifest}.toml env-1.10.12/
/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel -j2 --joblog results/prep/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3G /home1/mkolar/juliaPkg/julia-{}/bin/julia --startup-file=no code/prepare.jl {} > results/prep/julia-{}.log 2>&1' ::: 1.13.2 1.10.12
