#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=hdm106-objectives
#SBATCH --comment=20261026-hdm106-objective-suite-v1
#SBATCH --output=objective-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_CONSTRUCTION_SUITE="$PWD/objective-suite-v1"
test ! -e "$HDM_CONSTRUCTION_SUITE"
mkdir "$HDM_CONSTRUCTION_SUITE"
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
unset PARALLEL
"$HDM_PARALLEL" --jobs 4 --halt soon,fail=1 --joblog "$HDM_CONSTRUCTION_SUITE/joblog.tsv" --results "$HDM_CONSTRUCTION_SUITE/logs" srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$PWD/env-optional3-1.10" objective-v1-code/objective_suite.jl ::: {1..16}
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python objective-v1-code/objective_check.py
