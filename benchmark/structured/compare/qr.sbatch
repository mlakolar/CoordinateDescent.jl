#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:06:00
#SBATCH --job-name=cd80-qr
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=qr-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
export JULIA_PROJECT_ENV="$PWD/env-opt-v3-release"
mkdir -p reference-qr
parallel --jobs 3 --halt soon,fail=1 --joblog reference-qr/workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=1250M bash harness-qr/worker.sh {} R qr reference-qr > reference-qr/worker-{}.log 2>&1' ::: 8201 8202 8203
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
for stage in baseline-matched optimized-matched optimized-default; do
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project="$JULIA_PROJECT_ENV" harness-qr/check.jl inputs "accuracy-all/$stage" baseline-full reference-special reference-loose reference-qr "$stage"
done
python3 harness-qr/summarize.py . final-analysis-all --accuracy-root accuracy-all
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness-qr/plot.R . final-analysis-all
