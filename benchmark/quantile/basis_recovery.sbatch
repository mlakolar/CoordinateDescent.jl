#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-basis-recover-20261025
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=basis-recovery-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module load parallel/20240522
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=0
export J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
"$J" --project=. split_basis_validate.jl > split-basis-validation.log 2>&1
python3 basis_plan.py
mkdir -p basis-recovery-results basis-recovery-logs
fit_status=0
if ! parallel --colsep '\t' --jobs 2 --joblog basis-recovery-results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=8G --cpu-bind=cores "$J" --project=. basis_recovery.jl {1} {2} > basis-recovery-logs/{1}-{2}.log 2>&1' :::: basis-recovery-tasks.tsv; then fit_status=1; fi
python3 basis_merge.py
python3 summarize.py basis-completed basis-summary
python3 reconcile_bounds.py .
exit "$fit_status"
