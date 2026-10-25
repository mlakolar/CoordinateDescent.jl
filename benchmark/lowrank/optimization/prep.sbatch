#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cd81-opt-prep-20261025
#SBATCH --comment=20261025-cd81-optcomp
#SBATCH --output=prep-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH="/home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_PKG_PRECOMPILE_AUTO=0
mkdir -p env results inputs
cp /home1/mkolar/juliaPkg/runs/20261025T0910Z-cd81/env-final-1.13.2/{Project,Manifest}.toml env/
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --startup-file=no --project=env -e 'using Pkg; Pkg.add([PackageSpec(name="ProximalAlgorithms",version="0.7"),PackageSpec(name="ProximalOperators"),PackageSpec(name="ProximalCore"),PackageSpec(name="DelimitedFiles")]); Pkg.precompile(); Pkg.status()' > results/julia-setup.log 2>&1
module load python/3.11.9
python3 -m venv venv
venv/bin/python -m pip install --disable-pip-version-check numpy scipy pyproximal cvxpy threadpoolctl > results/python-setup.log 2>&1
venv/bin/python -m pip freeze > results/python-freeze.txt
venv/bin/python -c 'import numpy,scipy,pyproximal,cvxpy; print(numpy.__version__, scipy.__version__, pyproximal.__version__, cvxpy.__version__); numpy.show_config()' > results/python-config.txt
