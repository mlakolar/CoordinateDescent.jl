#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:19:00
#SBATCH --job-name=cd81-solvers-memory-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=memory-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
export PYTHON="$PWD/venv/bin/python"
module load rstats/4.5.2
unset PARALLEL
export CD81_PARALLEL_BIN="/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel"
# Submission requires the requested additional compute budget.
mkdir -p results/memory-pass
sha256sum code/memory-v3.sh code/updated_worker.jl code/resume_memory.py code/memory_summary.py > results/memory-pass/source-sha256.txt
"$PYTHON" code/resume_memory.py
"$CD81_PARALLEL_BIN" -j8 --colsep ' ' --joblog results/memory-pass/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/memory-v3.sh {1} {2} {3} > results/memory-pass/{1}-{3}.log 2>&1' :::: memory-tasks.txt
"$CD81_PARALLEL_BIN" -j2 --colsep ' ' --joblog results/memory-pass/large-joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/memory-v3.sh {1} {2} {3} > results/memory-pass/{1}-{3}.log 2>&1' :::: memory-large-tasks.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/memory_summary.py
"$PYTHON" code/collect.py
