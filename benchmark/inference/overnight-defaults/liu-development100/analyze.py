import csv,json,math,statistics
from pathlib import Path
D=Path('reports/liu-doubleml-development100');D.mkdir(exist_ok=True,parents=True)
summary=[];allrows=[];errors=[];notices=[]
def wilson(x,n):
 z=1.959963984540054;ph=x/n;den=1+z*z/n;mid=(ph+z*z/(2*n))/den;d=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den;return [mid-d,mid+d]
for score in ['instrument','nuisance_space']:
 method='doubleml_sparse_'+score;valid=[];missing=[];partial=0
 for rep in range(1,101):
  stem=Path('outputs/development/liu-published')/(str(rep)+'.'+method)
  if not Path(str(stem)+'.complete').exists():missing.append(rep)
  p=Path(str(stem)+'.csv');rs=list(csv.DictReader(p.open())) if p.exists() else []
  for r in rs:r.update(rep=rep,case='liu-published');allrows.append(r)
  for suffix,container in [('.errors.json',errors),('.warnings.json',notices)]:
   f=Path(str(stem)+suffix)
   if f.exists():
    obj=json.loads(f.read_text());container.extend(dict(rep=rep,method=method,**r) for r in obj)
  ok=len(rs)==2 and len({r['target'] for r in rs})==2 and all(r['converged'].lower()=='true' and float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) for r in rs)
  if ok:valid.append(rs)
  else:partial+=len(rs)
 for role in ['active','null']:
  rr=[next(r for r in rs if (float(r['truth'])!=0)==(role=='active')) for rs in valid]
  s=dict(method=method,role=role,valid_datasets=len(valid),planned_datasets=100,missing_markers=missing,partial_rows=partial)
  for key,fun in [('coverage',lambda r:float(r['lower'])<=float(r['truth'])<=float(r['upper'])),('rejection',lambda r:float(r['pvalue'])<.05),('length',lambda r:float(r['upper'])-float(r['lower'])),('seconds',lambda r:float(r['seconds']))]:
   v=[float(fun(r)) for r in rr];s[key]=statistics.mean(v) if v else None;s[key+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
   if key in ['coverage','rejection'] and v:s[key+'_wilson95_mc']=wilson(sum(v),len(v))
  summary.append(s)
checks=dict(planned_target_fits=400,finite_target_records=len(allrows),unique_records=len({(r['rep'],r['method'],r['target']) for r in allrows}),target_errors=len(errors),warning_entries=len(notices),note='Selected development expansion of favorable n1000,p201 regime; not fresh validation, not all-regime eligibility.')
for name,obj in [('checks',checks),('summary',summary),('errors',errors),('warnings',notices)]:(D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
with (D/'records.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
lines=['# Liu positive-control DoubleML development extension','','EMPIRICAL. The favorable published-regime control n1000,p201 is expanded from20to100development datasets. The native framework, declared learners, nested5x5cross-fitting, two scores and two requested target roles are unchanged. This is not the paper bootstrap-CI replica or independent validation. Failed other designs remain in the original twelve-case pilot.','','|Score|Role|Complete/planned|Coverage ± MCSE (%)|Rejection ± MCSE (%)|Length|','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['method']}|{r['role']}|{r['valid_datasets']}/100|{100*r['coverage']:.2f} ± {100*r['coverage_mcse']:.2f}|{100*r['rejection']:.2f} ± {100*r['rejection_mcse']:.2f}|{r['length']:.4f}|")
lines+=['','Boundary-aware Wilson Monte Carlo bounds are in summary.json; availability is separate from conditional coverage. No logistic default is changed and no large logistic campaign is eligible from this selected development extension alone.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(checks))
