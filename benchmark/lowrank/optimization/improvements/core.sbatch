#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --job-name=cd81-solvers-core-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=core-%j.out
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
mkdir -p results/core-v2
sha256sum code/adaptive-v2.jl code/analyze-v8.py code/collect-v10.py > results/core-v2/source-sha256.txt
CD81_PROFILE_ONLY=1 "$CD81_PARALLEL_BIN" -j5 --joblog results/core-v2/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env-1.13.2 code/adaptive-v2.jl {} > results/core-v2/rep-{}.log 2>&1' ::: 1 2 3 4 5
CD81_CORE_ONLY=1 srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/analyze-v8.py
"$PYTHON" code/collect-v10.py
