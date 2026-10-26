#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=hdm107-final-delivery
#SBATCH --comment=20261026-hdm107-final-delivery-v1
#SBATCH --output=final-delivery-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HDM_BASE="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python delivery-v1-code/convention_report.py
tar -czf reporting-delivery-v1.tar.gz completion-report-v2 completion-conventions-v1 \
 stat-main-v6/report stat-main-v6/checks stat-main-v6/fits/*.csv stat-main-v6/fits/*.toml stat-main-v6/fits/*.session.txt stat-main-v6/logs/*.log stat-main-v6/*joblog.tsv \
 runtime-main-v6/report runtime-main-v6/checks runtime-main-v6/fits/*.csv runtime-main-v6/fits/*.toml runtime-main-v6/fits/*.session.txt runtime-main-v6/logs/*.log runtime-main-v6/logs/*.resource.json runtime-main-v6/*joblog.tsv \
 env-comparison-1.10/Project.toml env-comparison-1.10/Manifest.toml \
 study-v6.tar completion-v2.tar stat-main-v6-13455468.out runtime-main-v6-13455471.out completion-report-v1-13455554.out completion-report-v2-13455573.out stat-main-v5-13455445.out runtime-main-v5-13455447.out
tar -czf worker-profiles-v1.tar.gz runtime-main-v6/fits/*-profiles
sha256sum reporting-delivery-v1.tar.gz worker-profiles-v1.tar.gz > delivery-v1.sha256
ls -lh reporting-delivery-v1.tar.gz worker-profiles-v1.tar.gz
