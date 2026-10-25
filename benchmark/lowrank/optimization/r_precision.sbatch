#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd81-opt-r-precision-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=r-precision-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export PYTHON="$PWD/venv/bin/python"
module load rstats/4.5.2
unset PARALLEL
CD81_PARALLEL_BIN="/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel"
mkdir -p results/r-coarse-timer results/r-precision
# Preserve superseded measurements and their adverse findings before replacing.
for d in results/main/*/r results/main/*/r-component; do
 case_id=$(basename "$(dirname "$d")")
 mkdir -p "results/r-coarse-timer/$case_id"
 cp -a "$d" "results/r-coarse-timer/$case_id/"
done
for d in results/memory/*/r_*; do
 case_id=$(basename "$(dirname "$d")")
 mkdir -p "results/r-coarse-timer/memory/$case_id"
 cp -a "$d" "results/r-coarse-timer/memory/$case_id/"
done
sha256sum code/r_worker_hires.R code/validate_r.py code/coverage_r.py > results/r-precision/source-sha256.txt
for rep in 1 2 3 4 5; do
 for case_name in small medium responses wide ill deficient; do
  printf '%s-%s\n' "$case_name" "$rep"
 done
done > r-precision-tasks.txt
"$CD81_PARALLEL_BIN" -j4 --joblog results/r-precision/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G Rscript code/r_worker_hires.R inputs/{} results/main/{}/r main > results/r-precision/{}.log 2>&1' :::: r-precision-tasks.txt
for rep in 1 2 3 4 5; do
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G Rscript code/r_worker_hires.R "inputs/responses-$rep" "results/main/responses-$rep/r-component" main component_only > "results/r-precision/responses-$rep-component.log" 2>&1
done
for rep in 1 2 3 4 5; do
 for case_name in responses wide; do
  for solver in r_rrr r_rrr_qr r_srrr; do
   printf '%s-%s %s\n' "$case_name" "$rep" "$solver"
  done
  if [ "$case_name" = wide ]; then printf '%s-%s r_component\n' "$case_name" "$rep"; fi
 done
done > r-memory-tasks.txt
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/r-precision/memory-joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G Rscript code/r_worker_hires.R inputs/{1} results/memory/{1}/{2} main memory {2} > results/r-precision/{1}-{2}.log 2>&1' :::: r-memory-tasks.txt
for rep in 1 2 3 4 5; do
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G Rscript code/r_worker_hires.R "inputs/responses-$rep" "results/memory/responses-$rep/r_component" main memory r_component > "results/r-precision/responses-$rep-component-memory.log" 2>&1
done
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/validate_r.py inputs results/main > results/main/validation.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/summarize.py results > results/summary.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/coverage_r.py results > results/coverage.log 2>&1
