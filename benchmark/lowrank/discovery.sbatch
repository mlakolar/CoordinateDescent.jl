#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=cd-lowrank
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 parallel/20240522
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
# Instantiate this environment before submission; set JULIA to its executable.
: "${JULIA:?set JULIA to the Julia executable}"
mkdir -p results/lowrank
parallel --halt soon,fail=1 -j4 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=benchmark/lowrank benchmark/lowrank/bench.jl {} results/lowrank > results/lowrank/worker-{}.log 2>&1' ::: 1 2 3 4 5
