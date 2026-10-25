#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --job-name=cd77-auto-speed-20261025
#SBATCH --output=bench-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/13.3.0 parallel/20240522 apptainer/1.3.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
prior=/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot
mkdir -p bench bench-data logs tmp
"$prior/venv/bin/python" source-v3/benchmark/inference/automatic-tuning/bench-prepare.py
parallel --colsep '\t' --joblog bench-joblog.tsv -j8 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M bash source-v3/benchmark/inference/automatic-tuning/bench-worker.sh {1} {2} {3} :::: bench-tasks.tsv
