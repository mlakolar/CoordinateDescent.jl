"""Supplement native convention comparisons from saved certified selections."""
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t

BASE=Path(os.environ['HDM_BASE']);STAT=BASE/'stat-main-v6'
OUT=BASE/'completion-conventions-v1';OUT.mkdir(exist_ok=False)
rows=defaultdict(dict)
for p in sorted((STAT/'checks').glob('*.csv')):
    if '.agreement.' in p.name:continue
    for r in csv.DictReader(p.open()):
        if r['engine']=='native' and r['method'] in ('paper','julia'):
            assert r['valid']=='True';rows[int(r['scenario']),int(r['seed'])][r['method']]=r
def estimate(values):
    a=np.array(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return dict(mean=None,se=None,lo=None,hi=None)
    mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)));delta=float(t.ppf(.975,len(a)-1)*se)
    return dict(mean=mean,se=se,lo=mean-delta,hi=mean+delta)
out=[]
for sc in range(1,17):
    rs=[v for (s,_),v in rows.items() if s==sc];assert len(rs)==1008
    r=dict(scenario=sc,inputs=len(rs),native_Julia_paper_selection_disagreements=sum(v['julia']['selected_ids']!=v['paper']['selected_ids'] for v in rs))
    for metric in ('fdp','power','selected','false'):
        for key,value in estimate([float(v['julia'][metric])-float(v['paper'][metric]) for v in rs]).items():r[f'{metric}_difference_{key}']=value
    out.append(r)
with (OUT/'native-convention-paired-differences.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(OUT/'summary.json').write_text(json.dumps(dict(rows=out,note='Native Julia-denominator rule minus native paper rule; all1008 inputs/scenario; fixed selected cutoff1e-6; paired pointwise Student-t MC intervals; original cohorts and all fits reused'),indent=2)+'\n')
print(json.dumps(out,indent=2))
