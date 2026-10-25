#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd77-decomp-detail-20261025
#SBATCH --output=detail-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd77-decomposition
python3 detail-v1.py
