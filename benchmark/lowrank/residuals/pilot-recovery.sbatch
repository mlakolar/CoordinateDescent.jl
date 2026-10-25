#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=hdm101-pilot3-1026
#SBATCH --comment=20261026-hdm101-study-v1-pilot-v3
#SBATCH --output=pilot-v3-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source environment.sh
# Preserve v2 failures; reuse the successful Julia and SciPy outputs.
for label in small-0 medium-0; do
  mv "results/pilot/$label/r" "results/pilot/$label/r-failed-v2"
done
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M Rscript -e 'capture.output(rrpack::rrr,file="results/pilot/installed-rrpack-rrr.R")'
printf '%s\n' small-0 medium-0 | "$HDM_PARALLEL" -j2 --joblog results/pilot/recovered-r-workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash worker.sh {} pilot r'
printf '%s\n' responses-0 wide-0 ill-0 deficient-0 | "$HDM_PARALLEL" -j2 --joblog results/pilot/remaining-workers.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash worker.sh {} pilot'
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M "$PYTHON" validate.py "$PWD" pilot
