#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$RUN/source"
export JULIA_DEPOT_PATH="$RUN/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SELECTION_RLIB="$RUN/Rlib"
module load gcc/13.3.0 apptainer parallel/20240522
mkdir -p "$RUN/golden" "$RUN/results"
if [[ ${FIRST_REP:-1} -eq 1 ]]; then
apptainer exec --cleanenv --env "OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,SELECTION_RLIB=$RUN/Rlib,R_LIBS_USER=$RUN/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3:/home1/mkolar/juliaPkg/Rlib-r4.5.3" /apps/containers/r/4.5/r-4.5.3.sif Rscript benchmark/selection/golden.R "$RUN/golden" > "$RUN/golden-R-$SLURM_JOB_ID.log" 2>&1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection benchmark/selection/golden.jl "$RUN/golden" > "$RUN/golden-Julia-$SLURM_JOB_ID.log" 2>&1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection -e 'using HighDimensionalModels, Test; include("test/selection.jl")' > "$RUN/unit-campaign-$SLURM_JOB_ID.log" 2>&1
fi
seq ${FIRST_REP:-1} ${LAST_REP:-7} | parallel --jobs 2 --halt soon,fail=1 --joblog "$RUN/parallel-$SLURM_JOB_ID.tsv" \
  "srun --exclusive --exact -N1 -n1 -c2 /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=benchmark/selection benchmark/selection/campaign.jl {} '$RUN/results' > '$RUN/worker-{}.log' 2>&1"
