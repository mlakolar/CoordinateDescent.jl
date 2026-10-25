"""Summarize saved development/validation fits on a Discovery compute node."""
from pathlib import Path
import csv,json,sys
from collections import defaultdict
import numpy as np
stage=sys.argv[1] if len(sys.argv)>1 else 'development';expected=int(sys.argv[2]) if len(sys.argv)>2 else 20
rows=[];errors=[];groups=defaultdict(list)
for f in Path('validation',stage).glob('*/*.csv'):
 case=f.parent.name
 for r in csv.DictReader(f.open()):
  r=dict(case=case,**r);rows.append(r);groups[case,r['method']].append(r)
for f in Path('validation',stage).glob('*/*.error'):errors.append(dict(file=str(f),error=f.read_text()))
for case in [f'{family}-{kind}' for family in ['gaussian','binomial'] for kind in ['moderate','correlated','benign','random']]:
 for method in ['cv','rldpe4_k50']:
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
Path('validation-'+stage+'-summary.json').write_text(json.dumps(dict(expected_per_case=expected,summary=summary,errors=errors),indent=2)+'\n')
with open('validation-'+stage+'-records.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
lines=['# Frozen-policy fresh validation','',f'{expected} planned datasets per case; summaries condition on converged output. Failures and fallbacks are retained in the JSON/raw files.','', '| Case | Method | Converged / expected | Active/null coverage | Type I | Power | Length active/null | Signed bias / SE |','|---|---|---:|---:|---:|---:|---:|---:|']
for d in summary:
 if not d['converged']:
  lines.append(f"| {d['case']} | {d['method']} | 0/{expected} | missing | — | — | — | — |")
  continue
 lines.append(f"| {d['case']} | {d['method']} | {d['converged']}/{expected} | {d['coverage_active']:.3f}/{d['coverage_null']:.3f} | {d['type1']:.3f} | {d['power']:.3f} | {d['length_active']:.3f}/{d['length_null']:.3f} | {d['signed_z_bias_active']:.3f} |")
lines+=['',f'Errors: {len(errors)}. Frozen candidate; holdout results must not be used to retune it.']
Path('validation-'+stage+'-RESULTS.md').write_text('\n'.join(lines)+'\n')
co=[]
for f in Path('validation',stage).glob('*/*.coefficients'):
 for row in csv.DictReader(f.open()):co.append(dict(case=f.parent.name,**row))
if co:
 with open('validation-'+stage+'-coefficients.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=co[0],lineterminator='\n');w.writeheader();w.writerows(co)
print('\n'.join(lines))

# Structural/numerical validation and dataset-level paired comparisons.
import math
expected_cases=[f'{family}-{kind}' for family in ['gaussian','binomial'] for kind in ['moderate','correlated','benign','random']]
index={(r['case'],int(r['rep']),r['method']):r for r in rows}
coindex={(r['case'],int(r['rep']),r['method'],int(r['target'])):r for r in co}
checks=dict(records=len(rows),unique_records=len(index),expected_records=expected*16,
 coefficients=len(co),unique_coefficients=len(coindex),expected_coefficients=expected*160,
 error_files=len(errors),all_converged=all(r['converged']=='true' for r in rows))
checks['positive_finite_se']=all(math.isfinite(float(r['se'])) and float(r['se'])>0 for r in co)
checks['finite_estimates']=all(math.isfinite(float(r['estimate'])) for r in co)
checks['normal_pvalue_consistency']=all(abs(math.erfc(abs(float(r['estimate'])/float(r['se']))/math.sqrt(2))-float(r['pvalue']))<1e-10 for r in co)
checks['candidate_noise_budget']=all(float(r['noise_ratio_max'])<=1.5*(1+1e-7) for r in rows if r['method']=='rldpe4_k50')
paired=[]
for case in expected_cases:
 d=dict(case=case)
 for metric in ['coverage_active','coverage_null','type1','power','length_active','length_null','seconds']:
  a=np.array([float(index[case,rep,'rldpe4_k50'][metric])-float(index[case,rep,'cv'][metric]) for rep in range(1,expected+1) if (case,rep,'cv') in index and (case,rep,'rldpe4_k50') in index])
  d[metric+'_difference']=float(a.mean()) if len(a) else None
  d[metric+'_difference_mcse']=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
 paired.append(d)
peaks=[]
for f in Path('resources').glob('validation-v5-*.txt'):
 try:peaks.append(dict(file=str(f),**{k:float(v) for k,v in (t.split('=') for t in f.read_text().split())}))
 except ValueError:pass
checks['resources']=dict(worker_processes=len(peaks),maximum_rss_kib=max((r['max_rss_kib'] for r in peaks),default=None),workers=peaks)
Path('validation-'+stage+'-checks.json').write_text(json.dumps(dict(checks=checks,paired=paired),indent=2)+'\n')
print('VALIDATION_CHECKS',json.dumps({k:v for k,v in checks.items() if k!='resources'}))
assert checks['records']==checks['unique_records']==checks['expected_records']
assert checks['coefficients']==checks['unique_coefficients']==checks['expected_coefficients']
assert checks['error_files']==0
assert all(checks[k] for k in ['all_converged','positive_finite_se','finite_estimates','normal_pvalue_consistency','candidate_noise_budget'])
