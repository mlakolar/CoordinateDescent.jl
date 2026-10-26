#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=30
#SBATCH --job-name=cd77-liu-doubleml-development100-20261026
#SBATCH --output=liu-doubleml-development100-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/source-v1
module load gcc/13.3.0 python/3.11.9 parallel/20240522 apptainer/1.3.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
apptainer exec --bind /home1 /apps/containers/r/4.5/r-4.5.3.sif Rscript ../liu-development100-v1/prepare.R
parallel --colsep '\t' --joblog liu-doubleml-development100-joblog.tsv -j32 srun --exclusive --exact -N1 -n1 -c1 --mem=900M python3 measure.py resources/liu-doubleml-development100-{2}-{5}.txt ../venv-doubleml/bin/python ../doubleml-v2/doubleml-logistic.py {1} {2} {3} {4} {5} :::: ../liu-development100-v1/tasks.tsv
python3 ../liu-development100-v1/analyze.py
cd ..
tar czf liu-doubleml-development100-final-v1.tar.gz source-v1/reports/liu-doubleml-development100 source-v1/resources/liu-doubleml-development100-* source-v1/liu-doubleml-development100-joblog.tsv liu-development100-inputs.txt
