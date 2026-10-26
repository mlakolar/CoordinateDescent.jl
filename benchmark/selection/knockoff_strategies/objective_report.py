"""Summarize saved construction diagnostics under Slurm; no new fitting."""
import csv,json,os
from pathlib import Path
import numpy as np
root=Path(os.environ['HDM_CONSTRUCTION_SUITE'])
rows=list(csv.DictReader((root/'independent-objectives.csv').open()))
for r in rows:
    for key in ('objective','stationarity','joint_min','root_error'):
        r[key]=float(r[key])
    r['returned']=r['returned']=='True';r['valid']=r['valid']=='True';r['case']=int(r['case'])
summary={'cohort':'16 fixed covariance matrices; no statistical evaluation data','groups':[], 'oracle_gaps':[], 'direct_gaps':[]}
for source in ('adapter','direct','CVXPY-CLARABEL'):
    for setting in sorted({r['setting'] for r in rows if r['source']==source}):
        group=[r for r in rows if r['source']==source and r['setting']==setting]
        valid=[r for r in group if r['returned'] and r['valid']]
        summary['groups'].append(dict(source=source,setting=setting,attempts=len(group),returned=sum(r['returned'] for r in group),
             feasible=len(valid),max_stationarity=max((r['stationarity'] for r in valid if np.isfinite(r['stationarity'])),default=None),
             max_root_error=max((r['root_error'] for r in valid if np.isfinite(r['root_error'])),default=None)))
for r in rows:
    if r['source']!='adapter' or r['setting']!='provisional':continue
    for source,destination in (('direct','direct_gaps'),('CVXPY-CLARABEL','oracle_gaps')):
        candidates=[s for s in rows if s['source']==source and s['case']==r['case'] and s['strategy']==r['strategy']
                    and s['setting']==('provisional' if source=='direct' else 'oracle')]
        if candidates:
            s=candidates[0]
            summary[destination].append(dict(case=r['case'],strategy=r['strategy'],other_status=s['error'],
                 valid=bool(r['valid'] and s['valid']),relative_objective_gap=float((r['objective']-s['objective'])/max(1,abs(s['objective'])))))
(root/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Fixed-matrix construction diagnostics','',
       'These are developmental numerical checks. They do not estimate statistical FDR/power or replicated runtime.',
       '', '| Source | Setting | Attempts | Returned | Feasible | Max full-joint gradient ratio |',
       '|---|---|---:|---:|---:|---:|']
for r in summary['groups']:
    lines.append(f"| {r['source']} | {r['setting']} | {r['attempts']} | {r['returned']} | {r['feasible']} | {r['max_stationarity']} |")
lines+=['','The full-joint gradient ratio divides each component by the sum of the corresponding positive inverse-matrix terms (at least one); it differs from the production max-term normalization. Original feasibility, not a backend perturbed matrix, is assessed. Every raw row, failed adapter error, oracle status and objective gap is retained. The conic oracle is a numerical comparator, not an exact proof.','',
        '| Matrix | Strategy | Adapter/oracle relative objective difference | Oracle status |','|---|---|---:|---|']
for r in summary['oracle_gaps']:
    lines.append(f"| {r['case']} | {r['strategy']} | {r['relative_objective_gap']:.6g} | {r['other_status']} |")
(root/'ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary))
