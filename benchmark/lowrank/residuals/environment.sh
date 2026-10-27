#!/bin/bash
module load rstats/4.5.2
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue101/depot:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
export BASE_ENV=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99/env-baseline-1.13
export CANDIDATE_ENV=/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue101/env-candidate-1.13
export PYTHON=/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/venv/bin/python
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/Rlib452
unset PARALLEL
export HDM_PARALLEL=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/parallel-20240522-qizrgyo/bin/parallel
