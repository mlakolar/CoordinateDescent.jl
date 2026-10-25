#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=cd78-tuning-compatibility-20261025
#SBATCH --output=tuning-compatibility-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-api/tuning-reference-v3
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261025-cd78-api/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - <<'PY'
import csv,json
from pathlib import Path
rows=list(csv.DictReader(Path('tied/results/tied-solutions.csv').open()))
bad=[r for r in rows if float(r['heldout_loss_difference'])>2e-6]
summary=dict(rows=len(rows),different_heldout_solutions=bad,max_objective_difference=max(float(r['objective_difference']) for r in rows),max_R_kkt=max(float(r['R_kkt']) for r in rows),max_R_abs_gap=max(abs(float(r['R_gap'])) for r in rows),max_API_kkt=max(float(r['API_kkt']) for r in rows),max_API_abs_gap=max(abs(float(r['API_gap'])) for r in rows))
Path('tied/results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --depwarn=error --project=compat-env -e 'using HighDimensionalModels,ProximalSolvers,ProximalBase,Test;@assert !ProximalSolvers._hasconjugate(ProximalBase.ConvolutionCheck(.5,.2));@assert Threads.nthreads()==2;include("HighDimensionalModels.jl/test/quantile_tuning.jl")' > compatibility-tests.log 2>&1
