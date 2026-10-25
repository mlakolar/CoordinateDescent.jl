#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=cd78-api-wide-control-20261025
#SBATCH --output=wide-accuracy-%j.out
set -euo pipefail
cd /home1/mkolar/juliaPkg/runs/20261025-cd78-api
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project=. wide-accuracy.jl > wide-accuracy.log 2>&1
python3 - <<'REPORT'
import csv,json,statistics as st
from pathlib import Path
r=list(csv.DictReader(open('wide-accuracy/paths.csv')));assert len(r)==4
summary=[]
for method in sorted({x['method'] for x in r}):
 rr=[x for x in r if x['method']==method]
 summary.append(dict(method=method,mean_total_s=st.mean(float(x['total_s']) for x in rr),max_kkt=max(float(x['max_kkt']) for x in rr),qualified_2e5=sum(x['qualified_2e5']=='true' for x in rr),qualified_2e6=sum(x['qualified_2e6']=='true' for x in rr)))
Path('wide-accuracy/summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
REPORT
