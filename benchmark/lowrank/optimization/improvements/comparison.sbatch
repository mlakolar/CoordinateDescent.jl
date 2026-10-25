#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --constraint=xeon-4116
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --job-name=cd81-solvers-compare-20261025
#SBATCH --comment=20261025-cd81-solvers
#SBATCH --output=compare-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export R_LIBS_USER="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452"
export JULIA="/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia"
export PYTHON="$PWD/venv/bin/python"
module load rstats/4.5.2
unset PARALLEL
CD81_PARALLEL_BIN="/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel"
mkdir -p results/main results/before results/memory results/adaptive
uname -a > results/platform.txt
lscpu >> results/platform.txt
sha256sum package/src/lowrank.jl package/ext/CoordinateDescentProximalAlgorithmsExt.jl code/*.jl code/*.py code/*.R > results/source-sha256.txt
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env-1.13.2 code/correctness.jl > results/correctness.log 2>&1
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/env -e 'using CoordinateDescent, ProximalAlgorithms' > results/before-precompile.log 2>&1
for rep in 1 2 3 4 5;do
 for name in small medium responses wide ill deficient;do
  for language in julia before python r;do printf '%s-%s %s\n' "$name" "$rep" "$language";done
 done
done > main-tasks.txt
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/main/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/worker.sh {1} {2} > results/main/{1}-{2}.log 2>&1' :::: main-tasks.txt
[ "${CD81_ORDINARY_ONLY:-0}" = 1 ] && exit 0
for rep in 1 2 3 4 5;do
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/worker.sh "responses-$rep" r-component > "results/main/responses-$rep-r-component.log" 2>&1
done
for rep in 1 2 3 4 5;do
 for name in responses wide;do
  for solver in cd_rank cd_nuclear cd_panoc ps_fista ps_admm pa_fista pa_panoc cd_sparse block_fista cd_sparse_api cd_sparse_fista cd_component component_fista;do
   printf '%s-%s julia %s\n' "$name" "$rep" "$solver"
  done
  for solver in before_rank before_nuclear before_sparse;do printf '%s-%s before %s\n' "$name" "$rep" "$solver";done
  for solver in scipy_rrr py_fista py_admm;do printf '%s-%s python %s\n' "$name" "$rep" "$solver";done
  for solver in r_rrr r_rrr_qr r_srrr;do printf '%s-%s r %s\n' "$name" "$rep" "$solver";done
  if [ "$name" = wide ];then printf '%s-%s r r_component\n' "$name" "$rep";fi
 done
done > memory-tasks.txt
"$CD81_PARALLEL_BIN" -j4 --colsep ' ' --joblog results/memory/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G bash code/memory.sh {1} {2} {3} > results/memory/{1}-{3}.log 2>&1' :::: memory-tasks.txt
for rep in 1 2 3 4 5;do
 srun --exclusive --exact -N1 -n1 -c1 --mem=8G bash code/memory.sh "responses-$rep" r r_component > "results/memory/responses-$rep-r_component.log" 2>&1
done
"$CD81_PARALLEL_BIN" -j4 --joblog results/adaptive/joblog.tsv \
 'srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$JULIA" --startup-file=no --project=env-1.13.2 code/adaptive.jl {} > results/adaptive/rep-{}.log 2>&1' ::: 1 2 3 4 5
srun --exclusive --exact -N1 -n1 -c1 --mem=2G "$PYTHON" code/analyze.py
"$PYTHON" code/collect.py
