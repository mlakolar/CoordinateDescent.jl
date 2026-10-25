#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --job-name=cd81-solvers-focused-memory-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=focused-memory-%j.out
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
mkdir -p results/focused-memory
sha256sum code/memory-v3.sh code/updated_worker.jl code/memory-summary-v2.py > results/focused-memory/source-sha256.txt
for solver in before_rank cd_rank cd_nuclear cd_panoc cd_sparse_api cd_sparse_fista;do
 case "$solver" in before_*) language=before;;*) language=julia;;esac
 for name in responses wide;do
  for rep in 1 2 3 4 5;do printf '%s-%s %s %s\n' "$name" "$rep" "$language" "$solver";done
 done
done > focused-memory-tasks.txt
"$CD81_PARALLEL_BIN" -j8 --colsep ' ' --joblog results/focused-memory/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/memory-v3.sh {1} {2} {3} > results/focused-memory/{1}-{3}.log 2>&1' :::: focused-memory-tasks.txt
CD81_EXPECTED_MEMORY=60 srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/memory-summary-v2.py
"$PYTHON" code/collect-v11.py
