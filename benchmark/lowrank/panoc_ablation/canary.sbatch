#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=hdm103-prototype-canary
#SBATCH --comment=20261026-hdm103-prototype-canary-v2
#SBATCH --output=prototype-canary-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
mkdir -p results/prototype-canary-v2
unset PARALLEL
HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
printf '%s\n' 1.10 1.13 | "$HDM_PARALLEL" -j2 --joblog results/prototype-canary-v2/tests.tsv 'srun --exclusive --exact -N1 -n1 -c1 --mem=3500M bash prototype-v2-code/canary-worker.sh {}'
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env-candidate2-1.13 prototype-v2-code/canary_export.jl
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python prototype-v2-code/canary_check.py
