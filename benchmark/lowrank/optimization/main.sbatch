#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --job-name=cd81-opt-main-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=main-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
export PYTHON="$PWD/venv/bin/python"
CD81_PARALLEL_BIN="/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel"
unset PARALLEL
module load rstats/4.5.2
sha256sum code/*.jl code/*.py code/*.R code/*.sh code/*.sbatch > results/source-sha256.txt
uname -a > results/platform.txt
lscpu >> results/platform.txt
mkdir -p results/main results/memory
for rep in 1 2 3 4 5; do
  for case_name in small medium responses wide ill deficient; do
    for language in julia python r; do
      printf '%s-%s %s\n' "$case_name" "$rep" "$language"
    done
  done
done > main-tasks.txt
set +e
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/main/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker.sh {1} {2} main > results/main/{1}-{2}.log 2>&1' :::: main-tasks.txt
ordinary_status=$?
set -e
# rrpack materializes a 2.4 GB response design at q=1000: isolated8GB steps.
for rep in 1 2 3 4 5; do
 if [ -s "results/main/responses-$rep/r-component/r-component.csv" ]; then continue; fi
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/worker.sh "responses-$rep" r-component main > "results/main/responses-$rep-r-component.log" 2>&1
done
for rep in 1 2 3 4 5; do
 for case_name in responses wide; do
  for solver in cd_rank cd_nuclear ps_fista ps_admm pa_fista pa_panoc cd_sparse block_fista cd_component component_fista; do
   printf '%s-%s julia %s\n' "$case_name" "$rep" "$solver"
  done
  for solver in scipy_rrr py_fista py_admm; do
   printf '%s-%s python %s\n' "$case_name" "$rep" "$solver"
  done
  for solver in r_rrr r_rrr_qr r_srrr; do printf '%s-%s r %s\n' "$case_name" "$rep" "$solver"; done
  if [ "$case_name" = wide ]; then printf '%s-%s r r_component\n' "$case_name" "$rep"; fi
 done
done > memory-tasks.txt
set +e
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/memory/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/memory.sh {1} {2} {3} > results/memory/{1}-{3}.log 2>&1' :::: memory-tasks.txt
memory_status=$?
set -e
for rep in 1 2 3 4 5; do
 if [ -s "results/memory/responses-$rep/r_component/memory.csv" ]; then continue; fi
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/memory.sh "responses-$rep" r r_component > "results/memory/responses-$rep-r_component.log" 2>&1
done
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/validate.py inputs results/main > results/main/validation.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/summarize.py results > results/summary.log 2>&1
test "$ordinary_status" -eq 0
test "$memory_status" -eq 0
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/coverage.py results
