#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hdm107-stopping-probe
#SBATCH --comment=20261026-hdm107-stopping-probe-v1
#SBATCH --output=stopping-probe-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HDM_BASE="$PWD" HDM_CODE="$PWD/stopping-v1-code" HDM_STUDY="$PWD/runtime-pilot-v1"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog stopping-probe-v1-joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$HDM_BASE/env-comparison-1.10" "$HDM_CODE/stopping_probe.jl" {} > "$HDM_BASE/stopping-probe-v1-{}.log" 2>&1' ::: $(seq 1 4)
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python "$HDM_CODE/stopping_probe_check.py" {} > "$HDM_BASE/stopping-probe-check-v1-{}.log" 2>&1' ::: $(seq 1 4)
