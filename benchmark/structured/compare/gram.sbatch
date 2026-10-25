#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:06:00
#SBATCH --job-name=cd80-gram
#SBATCH --comment=20261025-cd80-competitive
#SBATCH --output=gram-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522 apptainer
mkdir -p optimized-gram-matched optimized-gram-default optimized-gram-profile
parallel --jobs 3 --halt soon,fail=1 --joblog gram-workers.tsv \
  'srun --exclusive --exact -N1 -n1 -c1 --mem=1250M bash harness-gram/gram-worker.sh {} > gram-worker-{}.log 2>&1' ::: 8201 8202 8203
old=/home1/mkolar/juliaPkg/runs/20261025-cd80-structured
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_DEPOT_PATH="$PWD/depot-opt-v4-release:$old/depot-release:$old/depot:/home1/mkolar/juliaPkg/depot:"
mkdir -p accuracy-final/baseline-matched
cp accuracy-all/baseline-matched/accuracy.tsv accuracy-final/baseline-matched/accuracy.tsv
for stage in optimized-matched optimized-default; do
  override="${stage/optimized-/optimized-gram-}"
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-opt-v4-release harness-gram/check.jl inputs "accuracy-final/$stage" baseline-full reference-special reference-loose reference-qr "$stage" "$override"
done
python3 harness-gram/summarize.py . final-report --accuracy-root accuracy-final
apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript harness-gram/plot.R . final-report
