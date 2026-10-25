#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd81-opt-pilot-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=pilot-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
export PYTHON="$PWD/venv/bin/python"
module load rstats/4.5.2
srun --exclusive --exact -N1 -n1 -c1 --mem=3G "$PYTHON" code/common.py inputs > results/generate.log 2>&1
mkdir -p results/pilot
for case_id in medium-1 medium-2 wide-1 wide-2; do
  for language in julia python r; do
    printf '%s %s\n' "$case_id" "$language"
  done
done > pilot-tasks.txt
/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel -j4 --colsep ' ' --joblog results/pilot/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker.sh {1} {2} pilot > results/pilot/{1}-{2}.log 2>&1' :::: pilot-tasks.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/validate.py inputs results/pilot --cvxpy > results/pilot/validation.log 2>&1
