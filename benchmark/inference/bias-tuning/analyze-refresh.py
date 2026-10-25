"""Combine validated cached competitor fits with current-default Julia fits."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
prior=Path('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/pilot-records.csv')
rows=list(csv.DictReader(prior.open()))
rows=[r for r in rows if not(r['engine'] in ['ours','oursqp'] and r['mode']=='default')]
new=[]
for f in Path('default-refresh').glob('*/*.csv'):new.extend(csv.DictReader(f.open()))
assert len(new)==200
assert len({(r['scenario'],r['rep'],r['engine'],r['mode']) for r in new})==200
rows+=new
out=Path('final-report');out.mkdir(exist_ok=True)
fields=sorted(set(k for r in rows for k in r))
with (out/'current-pilot-records.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
groups=defaultdict(list)
for r in rows:groups[r['scenario'],r['engine'],r['mode']].append(r)
summary=[]
for (case,engine,mode),rs in sorted(groups.items()):
 good=[r for r in rs if r['status']=='ok'];d=dict(case=case,engine=engine,mode=mode,records=len(rs),successful=len(good),failed=sum(r['status'] not in ['ok','unsupported'] for r in rs),unsupported=sum(r['status']=='unsupported' for r in rs))
 for key in ['active_coverage','null_coverage','null_rejection','active_rejection','active_length','null_length','fit_seconds','max_rss_kib']:
  a=np.array([float(r[key]) for r in good if r.get(key) not in [None,'']]);a=a[np.isfinite(a)]
  d[key]=float(a.mean()) if len(a) else None;d[key+'_mcse']=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None;d[key+'_median']=float(np.median(a)) if len(a) else None;d[key+'_max']=float(a.max()) if len(a) else None
 summary.append(d)
(out/'current-pilot-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Current defaults on the paired competitor pilot','',
 'EMPIRICAL. Twenty datasets per original design. Current Julia default fits replace the historical default rows on exactly the same saved inputs; matched Julia rows and competitor rows reuse the previously validated fits. The frozen restricted-LDPE candidate is labeled experimental. These legacy pilot inputs are separate from the new 200-dataset-per-design validation; this refresh does not choose or retune the candidate.','',
 'The requested task has ten targets. Julia and SIHR compute those targets, while hdi/EconML/sslasso compute all500 slopes internally. Full-pipeline timing here is not an equal-work solver benchmark; use the separately matched500-slope/row benchmark for that comparison. New Julia memory is measured for combined worker processes and is not imputed to individual policy rows. Old per-process competitor memory remains in the records.','',
 '| Case | Engine | Tuning | Valid/20 | Active coverage ± MCSE | Null coverage | Type I | Power | Length active/null | Median warm seconds |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for d in summary:
 if not d['successful']:
  lines.append(f"| {d['case']} | {d['engine']} | {d['mode']} | 0/20 | unsupported/failed | — | — | — | — | — |")
 else:
  lines.append(f"| {d['case']} | {d['engine']} | {d['mode']} | {d['successful']}/20 | {100*d['active_coverage']:.1f}% ± {100*d['active_coverage_mcse']:.1f} pp | {100*d['null_coverage']:.1f}% | {100*d['null_rejection']:.1f}% | {100*d['active_rejection']:.1f}% | {d['active_length']:.3f}/{d['null_length']:.3f} | {d['fit_seconds_median']:.3f} |")
lines+=['',
 'Current Julia defaults now select initial penalties automatically for both response families; the historical required-GLM-lambda note applies only to the superseded pilot rows. Matched rows keep the explicit common penalties. Expert restricted LDPE is not a competitor-matched projection rule.',
 '', 'Native hdi binomial matched preprocessing remains unsupported. EconML and sslasso are Gaussian-only. SIHR can adjust supplied constraints and retains a different target pipeline. sslasso intervals include bias padding and need not be dual to the returned normal p-values. See the original pilot report for warnings and reference-code corrections.','',
 f'Current-default refresh: {len(new)} fits; {sum(r["status"]=="ok" for r in new)} successful. Failures, if any, remain in the records; no numerical default is silently tightened for this comparison.']
(out/'CURRENT_COMPARISON.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
assert all(r['status']=='ok' for r in new)
