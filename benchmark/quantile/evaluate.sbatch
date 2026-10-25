#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-main-20261025
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=evaluate-%j.out
set -euo pipefail
export QUANTILE_RUN_ROOT="${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module load parallel/20240522 gurobi/12.0.3
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=1
export J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
"$J" --project=. validate.jl > validation-v3.log 2>&1
mkdir -p main-results main-logs
lscpu > main-results/cpu.txt
scontrol show job "$SLURM_JOB_ID" > main-results/allocation.txt
# Four workers total, each bound to one allocated core and at most4GiB.
# srun steps inherit the allocation's two nodes, choosing one per worker.
parallel --colsep '\t' --jobs 4 --joblog main-results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores timeout 900 "$J" --project=. worker.jl {1} {2} 2 {3} main-results > main-logs/{1}-{2}.log 2>&1' :::: tasks.tsv

python3 summarize.py main-results summary
"$J" --project=. density.jl density-results > density.log 2>&1
