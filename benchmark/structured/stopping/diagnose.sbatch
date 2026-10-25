#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=cd80-stopping-diagnose
#SBATCH --comment=20261025-cd80-stopping
#SBATCH --output=diagnose-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="$prior/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
mkdir -p diagnostics
parallel --jobs 3 --halt soon,fail=1 --joblog diagnostics/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project='"$prior"'/env-opt-v4-release harness/diagnose.jl {} '"$prior"' diagnostics > diagnostics/worker-{}.log 2>&1' ::: 8201 8202 8203
