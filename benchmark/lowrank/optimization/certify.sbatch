#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd81-opt-certify-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=certify-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
export PYTHON="$PWD/venv/bin/python"
module load rstats/4.5.2
mkdir -p results/certify
srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker.sh wide-2 julia pilot > results/certify/wide-2-julia.log 2>&1 &
julia_pid=$!
srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker.sh wide-2 r pilot > results/certify/wide-2-r.log 2>&1 &
r_pid=$!
wait "$julia_pid"
wait "$r_pid"
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/validate.py inputs results/pilot --cvxpy > results/certify/validation.log 2>&1
