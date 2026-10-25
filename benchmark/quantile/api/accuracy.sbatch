#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --job-name=cd78-api-accuracy-20261025
#SBATCH --output=accuracy-%j.out
set -euo pipefail
cd "${QUANTILE_RUN_ROOT:-$SLURM_SUBMIT_DIR}"
module purge
module load r/4.5.3
export R_LIBS_USER=/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/Rlib:/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. accuracy.jl > accuracy.log 2>&1
python3 - <<'PY'
import csv,json,statistics as st
from pathlib import Path
r=list(csv.DictReader(open('accuracy/paths.csv')));groups={}
for row in r:groups.setdefault((row['case'],row['method']),[]).append(row)
s=[]
for (case,method),rows in sorted(groups.items()):
 s.append(dict(case=case,method=method,reps=len(rows),mean_total_s=st.mean(float(x['total_s']) for x in rows),max_kkt=max(float(x['max_kkt']) for x in rows),qualified_2e5=sum(x['qualified_2e5']=='true' for x in rows),qualified_2e6=sum(x['qualified_2e6']=='true' for x in rows)))
Path('accuracy/summary.json').write_text(json.dumps(s,indent=2));assert len(r)==20
print('Accuracy sensitivity20observations complete; one input per shape, no input-level CI.')
PY
