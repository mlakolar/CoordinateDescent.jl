#!/bin/bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load rstats/4.5.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452
exec Rscript --vanilla "${HDM_HARNESS:-full-v2}"/r_worker.R "$@"
