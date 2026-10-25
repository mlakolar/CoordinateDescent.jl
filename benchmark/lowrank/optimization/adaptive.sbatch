#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd81-opt-adaptive-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=adaptive-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export PYTHON="$PWD/venv/bin/python"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
module load rstats/4.5.2
mkdir -p results/adaptive
sha256sum code/adaptive* > results/adaptive/source-sha256.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/adaptive_inputs.py inputs
for rep in 1 2 3 4 5; do
 printf '%s julia\n%s r\n' "$rep" "$rep"
done > adaptive-tasks.txt
/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel -j2 --colsep ' ' --joblog results/adaptive/joblog.tsv \
 'if [ {2} = julia ]; then srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env code/adaptive.jl {1}; else srun --exclusive --exact -N1 -n1 -c1 --mem=2G Rscript code/adaptive.R {1}; fi > results/adaptive/{1}-{2}.log 2>&1' :::: adaptive-tasks.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/adaptive_validate.py > results/adaptive/validation.log 2>&1
