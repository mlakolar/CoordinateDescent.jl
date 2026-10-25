#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd80-stopping-validate
#SBATCH --comment=20261025-cd80-stopping
#SBATCH --output=validate-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
prior=/home1/mkolar/juliaPkg/runs/20261025-cd80-competitive
for version in release lts; do
  mkdir -p "env-gap-$version"
  for f in Project.toml Manifest.toml; do
    sed "s|$prior/source-opt-v4|$PWD/source-gap-v1|g" "$prior/env-opt-v4-$version/$f" > "env-gap-$version/$f"
  done
done
mkdir -p validation calibration-warmfix
parallel --jobs 3 --halt soon,fail=1 --joblog validation/workers.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3800M bash harness-validation/validate-worker.sh {} > validation/{}.log 2>&1' ::: release lts warmfix
