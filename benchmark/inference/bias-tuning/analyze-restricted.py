"""Summarize saved development/validation fits on a Discovery compute node."""
from pathlib import Path
import csv,json,sys
from collections import defaultdict
import numpy as np
stage=sys.argv[1] if len(sys.argv)>1 else 'development';expected=int(sys.argv[2]) if len(sys.argv)>2 else 20
rows=[];errors=[];groups=defaultdict(list)
for f in Path('restricted',stage).glob('*/*.csv'):
 case=f.parent.name
 for r in csv.DictReader(f.open()):
  r=dict(case=case,**r);rows.append(r);groups[case,r['method']].append(r)
for f in Path('restricted',stage).glob('*/*.error'):errors.append(dict(file=str(f),error=f.read_text()))
for case in [f'{family}-{kind}' for family in ['gaussian','binomial'] for kind in ['moderate','correlated','benign','random']]:
 for method in ['cv','ldpe','ldpe_k50','rldpe4','rldpe4_k50']:
  groups[case,method]
summary=[]
for (case,method),rs in sorted(groups.items()):
 good=[r for r in rs if r['converged']=='true']
 d=dict(case=case,method=method,expected=expected,records=len(rs),converged=len(good),initial_statuses={status:sum(r['initial_status']==status for r in rs) for status in sorted(set(r['initial_status'] for r in rs))})
 for key in ['coverage_active','coverage_null','type1','power','length_active','length_null','signed_z_bias_active','sigma','bias_factor_max','noise_ratio_max','initial_size','seconds']:
  a=np.array([float(r[key]) for r in good]);valid=a[np.isfinite(a)]
  d[key]=float(valid.mean()) if len(valid) else None
  d[key+'_median']=float(np.median(valid)) if len(valid) else None
  d[key+'_mcse']=float(valid.std(ddof=1)/np.sqrt(len(valid))) if len(valid)>1 else None
 summary.append(d)
Path('restricted-'+stage+'-summary.json').write_text(json.dumps(dict(expected_per_case=expected,summary=summary,errors=errors),indent=2)+'\n')
with open('restricted-'+stage+'-records.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
lines=['# Bias-aware projection development','',f'{expected} planned datasets per case; summaries condition on converged output. Failures and fallbacks are retained in the JSON/raw files.','', '| Case | Method | Converged / expected | Active/null coverage | Type I | Power | Length active/null | Signed bias / SE |','|---|---|---:|---:|---:|---:|---:|---:|']
for d in summary:
 if not d['converged']:
  lines.append(f"| {d['case']} | {d['method']} | 0/{expected} | missing | — | — | — | — |")
  continue
 lines.append(f"| {d['case']} | {d['method']} | {d['converged']}/{expected} | {d['coverage_active']:.3f}/{d['coverage_null']:.3f} | {d['type1']:.3f} | {d['power']:.3f} | {d['length_active']:.3f}/{d['length_null']:.3f} | {d['signed_z_bias_active']:.3f} |")
lines+=['',f'Errors: {len(errors)}. The holdout seed namespace remains untouched during development.']
Path('restricted-'+stage+'-RESULTS.md').write_text('\n'.join(lines)+'\n')
co=[]
for f in Path('restricted',stage).glob('*/*.coefficients'):
 for row in csv.DictReader(f.open()):co.append(dict(case=f.parent.name,**row))
if co:
 with open('restricted-'+stage+'-coefficients.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=co[0],lineterminator='\n');w.writeheader();w.writerows(co)
print('\n'.join(lines))
