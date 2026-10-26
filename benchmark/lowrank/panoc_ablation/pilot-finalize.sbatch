#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=hdm103-pilot-report
#SBATCH --comment=20261026-hdm103-pilot-report-v1
#SBATCH --output=pilot-report-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 code/pilot_finalize.py
