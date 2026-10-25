#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --job-name=cd78-basis-20261025
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --output=basis-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module load parallel/20240522
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BENCH_GUROBI=0
export J=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
fit_status=0
if "$J" --project=. basis_validate.jl > basis-validation.log 2>&1; then
mkdir -p basis-results basis-logs
if ! parallel --colsep '\t' --jobs 3 --joblog basis-results/parallel.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores "$J" --project=. basis_worker.jl {1} {2} basis-results > basis-logs/{1}-{2}.log 2>&1' :::: basis-tasks.tsv; then fit_status=1; fi
python3 summarize.py basis-results basis-summary
else
fit_status=1
fi
python3 reconcile_bounds.py .

exit "$fit_status"
