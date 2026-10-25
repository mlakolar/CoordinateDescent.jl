import csv,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
stage=sys.argv[1];root=Path('outputs')/stage
cases=['binomial-moderate','binomial-correlated','binomial-benign','binomial-random','gaussian-moderate','gaussian-benign']
counts={c:(1 if stage=='canary' else 100 if c.startswith('gaussian') else 200) for c in cases}
rows=[];errors=[];missing=[]
for c,count in counts.items():
 for rep in range(1,count+1):
  stem=root/c/str(rep)
  if not stem.with_suffix('.complete').exists():missing.append([c,rep])
  if stem.with_suffix('.error').exists():errors.append([c,rep,stem.with_suffix('.error').read_text()])
  if stem.with_suffix('.csv').exists():
   for r in csv.DictReader(stem.with_suffix('.csv').open()):r['case']=c;rows.append(r)
expected=sum(counts[c]*(100 if c=='gaussian-benign' else 80 if c.startswith('gaussian') else 200 if c=='binomial-benign' else 160) for c in cases)
keys=[(r['case'],r['rep'],r['profile'],r['weights'],r['initial'],r['target']) for r in rows]
checks={'expected_rows':expected,'rows':len(rows),'unique_rows':len(set(keys)),'missing':missing,'errors':errors,'all_converged':all(r['converged']=='true' for r in rows),'finite_positive_se':all(math.isfinite(float(r['se'])) and float(r['se'])>0 for r in rows),'max_identity_error':max((abs(float(r['identity_error'])) for r in rows),default=0),'max_score_intercept_orthogonality':max((float(r['score_orthogonality']) for r in rows),default=0)}
groups=defaultdict(lambda:defaultdict(list))
for r in rows:
 groups[(r['case'],r['profile'],r['weights'],r['initial'],'active' if float(r['truth']) else 'null')][int(r['rep'])].append(r)
def mean(x):return statistics.mean(x)
def mc(x):return statistics.stdev(x)/math.sqrt(len(x)) if len(x)>1 else None
summaries=[]
for key,reps in sorted(groups.items()):
 metrics=defaultdict(list)
 for rep,rr in reps.items():
  metrics['coverage'].append(mean([float(r['covered']=='true') for r in rr]))
  metrics['rejection'].append(mean([float(r['rejected']=='true') for r in rr]))
  metrics['interval_length'].append(mean([2*1.959963984540054*float(r['se']) for r in rr]))
  metrics['lower_endpoint_fraction'].append(mean([float(r['endpoint']=='lower') for r in rr]))
  metrics['upper_endpoint_fraction'].append(mean([float(r['endpoint']=='upper') for r in rr]))
  for name in ['stochastic','linear_bias','nonlinear_remainder']:
   metrics[name+'_signed_z'].append(mean([float(r[name])/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr]))
  metrics['total_signed_z'].append(mean([(float(r['estimate'])-float(r['truth']))/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr]))
  metrics['true_variance_se_ratio'].append(mean([float(r['true_variance_se'])/float(r['se']) for r in rr]))
  metrics['robust_se_ratio'].append(mean([float(r['robust_se'])/float(r['se']) for r in rr]))
 summary=dict(zip(['case','profile','weights','initial','target_class'],key));summary['datasets']=len(reps)
 for name,v in metrics.items():summary[name]=mean(v);summary[name+'_mcse']=mc(v)
 summaries.append(summary)
# All changes relative to the base grid on the same dataset and target.
index={k:r for k,r in zip(keys,rows)};pairs=defaultdict(lambda:defaultdict(list))
for r in rows:
 if r['profile']=='base':continue
 k=(r['case'],r['rep'],'base',r['weights'],r['initial'],r['target'])
 if k not in index:continue
 b=index[k];g=(r['case'],r['profile'],r['weights'],r['initial'],'active' if float(r['truth']) else 'null')
 pairs[g][r['rep']].append({'estimate_delta_over_base_se':(float(r['estimate'])-float(b['estimate']))/float(b['se']),'se_ratio':float(r['se'])/float(b['se']),'coverage_difference':float(r['covered']=='true')-float(b['covered']=='true')})
paired=[]
for k,reps in sorted(pairs.items()):
 d=dict(zip(['case','profile','weights','initial','target_class'],k));d['datasets']=len(reps)
 for name in ['estimate_delta_over_base_se','se_ratio','coverage_difference']:
  v=[mean([r[name] for r in rr]) for rr in reps.values()];d[name]=mean(v);d[name+'_mcse']=mc(v)
 paired.append(d)
out=Path('reports')/stage;out.mkdir(parents=True,exist_ok=True)
(out/'checks.json').write_text(json.dumps(checks,indent=2));(out/'summary.json').write_text(json.dumps(summaries,indent=2));(out/'paired.json').write_text(json.dumps(paired,indent=2))
with (out/'coefficients.csv').open('w') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(checks,indent=2))
if missing or errors or len(rows)!=expected or len(set(keys))!=expected or not checks['all_converged'] or not checks['finite_positive_se'] or checks['max_identity_error']>1e-8:sys.exit(1)
