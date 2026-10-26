#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=hdm106-sampler-reference
#SBATCH --comment=20261026-hdm106-sampler-reference-v1
#SBATCH --output=sampler-reference-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/depot:/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot:/home1/mkolar/juliaPkg/depot:"
export HDM_REFERENCE="$PWD/sampler-reference-v1"
test ! -e "$HDM_REFERENCE"
mkdir "$HDM_REFERENCE"
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --startup-file=no --project="$PWD/env-optional3-1.10" sampler-reference-v1-code/sampler_reference.jl
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M apptainer exec --cleanenv --env "HDM_REFERENCE=$HDM_REFERENCE,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" /apps/containers/r/4.5/r-4.5.3.sif Rscript "$PWD/sampler-reference-v1-code/sampler_reference.R"
srun --exclusive --exact -N1 -n1 -c1 --mem=3500M /home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python sampler-reference-v1-code/sampler_reference_check.py
