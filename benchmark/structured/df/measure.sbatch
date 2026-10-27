#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=hdm99-measure-1026
#SBATCH --comment=20261026-hdm-overnight-issue99-measure-v1
#SBATCH --output=measure-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
module load parallel/20240522
mkdir -p results
parallel --jobs 7 --halt soon,fail=1 --joblog results/workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash worker.sh {}' ::: 1 2 3 4 5 6 7
srun --exclusive --exact -N1 -n1 -c1 --mem=3800M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-candidate-1.13 validate.jl
srun --exclusive --exact -N1 -n1 -c1 --mem=3800M python3 summarize.py
