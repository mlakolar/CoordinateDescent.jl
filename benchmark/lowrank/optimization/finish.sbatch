#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd81-opt-finish-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=finish-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export PYTHON="$PWD/venv/bin/python"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
module load rstats/4.5.2
unset PARALLEL
mkdir -p results/finish results/adaptive-r-coarse
for rep in 1 2 3 4 5; do
 cp -a "results/adaptive/medium-$rep/r" "results/adaptive-r-coarse/medium-$rep-r"
 srun --exclusive --exact -N1 -n1 -c1 --mem=2G Rscript code/adaptive_hires.R "$rep" > "results/finish/adaptive-r-$rep.log" 2>&1
done
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/adaptive_validate.py > results/adaptive/validation.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env code/rank_core.jl > results/finish/rank-core.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/rank_core_summary.py > results/rank-core/validation.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G Rscript -e 'invisible(crossprod(matrix(1,64,64))); print(R.version.string); print(extSoftVersion()); print(Sys.getenv(c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"))); print(grep("^(Threads|Cpus_allowed_list):",readLines("/proc/self/status"),value=TRUE))' > results/r-configuration.txt
sha256sum code/adaptive_hires.R code/rank_core.jl code/rank_core_summary.py > results/finish/source-sha256.txt
sha256sum inputs/*/X.bin inputs/*/Y.bin > results/input-sha256.txt
