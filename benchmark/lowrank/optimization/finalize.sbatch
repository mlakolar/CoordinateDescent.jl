#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --job-name=cd81-opt-report-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=report-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
srun --exclusive --exact -N1 -n1 -c1 venv/bin/python code/finalize.py
