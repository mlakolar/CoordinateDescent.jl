#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:03:00
#SBATCH --job-name=cd81-opt-calibrate-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=calibrate-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env code/calibrate.jl > results/calibration.log 2>&1
