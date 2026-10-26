#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=300
#SBATCH --job-name=cd77-gaussian-native-core1000-20261026
#SBATCH --output=gaussian-native-core1000-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/source-v1
module load gcc/13.3.0 parallel/20240522 apptainer/1.3.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/depot:/home1/mkolar/juliaPkg/depot:
parallel --colsep '\t' --joblog gaussian-native-full-joblog.tsv -j64 srun --exclusive --exact -N1 -n1 -c1 --mem=1800M python3 measure.py resources/gaussian-native-full-{1}-{2}-{3}.txt bash ../gaussian-native-full-v1/gaussian-native-full-worker.sh {1} {2} {3} :::: ../gaussian-native-full-v1/gaussian-native-full-tasks.tsv
