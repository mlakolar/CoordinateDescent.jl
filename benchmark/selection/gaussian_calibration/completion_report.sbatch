#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=hdm107-completion-report
#SBATCH --comment=20261026-hdm107-completion-report-v2
#SBATCH --output=completion-report-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HDM_BASE="$PWD" HDM_STUDY="$PWD/stat-main-v6" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python completion-v2-code/completion_report.py
