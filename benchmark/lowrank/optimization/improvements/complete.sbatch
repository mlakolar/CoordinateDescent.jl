#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --job-name=cd81-solvers-complete-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=complete-%j.out
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
mkdir -p results/recovery
lscpu > results/recovery/platform.txt
sha256sum code/worker-v3.sh code/memory-v3.sh code/recover_tasks.py code/remaining.sh code/analyze-v3.py code/collect-v3.py complete.sbatch > results/recovery/source-sha256.txt
"$PYTHON" code/recover_tasks.py > results/recovery/selection.log
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/recovery/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker-v3.sh {1} {2} > results/recovery/{1}-{2}.log 2>&1' :::: recovery-tasks.txt
source code/remaining.sh
