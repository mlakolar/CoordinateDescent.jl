"""Compute-node all-slope automatic-policy timing and target-list stability."""
import csv,json,math,statistics
from pathlib import Path
cases=['gaussian-moderate','gaussian-correlated','gaussian-benign'];rows=[];fits=[];clusters=[];errors=[];expected=22000;maxdiff=0.;npaired=0
for c in cases:
 for rep in range(1,21):
  stem=Path('outputs/gaussian-all-target-time')/c/f'{rep}.ours_all_warm';rs=Path('resources')/f'gaussian-all-target-time-{c}-{rep}.txt'
  meta=dict(l.split('=',1) for l in Path(str(stem)+'.meta').read_text().splitlines() if '=' in l) if Path(str(stem)+'.meta').exists() else {}
  rr=list(csv.DictReader(Path(str(stem)+'.csv').open())) if Path(str(stem)+'.csv').exists() else []
  er=Path(str(stem)+'.error');diagnostics=list(csv.DictReader(Path(str(stem)+'.diagnostics.csv').open())) if Path(str(stem)+'.diagnostics.csv').exists() else []
  if er.exists():errors.append(dict(case=c,rep=rep,error=er.read_text()))
  resource=dict(l.split('=',1) for l in rs.read_text().splitlines() if '=' in l) if rs.exists() else {}
  n=100 if c=='gaussian-benign' else 500
  ok=meta.get('status')=='ok' and not er.exists() and len(rr)==n and len({r['target'] for r in rr})==n and all(r['converged'].lower()=='true' and float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) for r in rr)
  f=dict(case=c,rep=rep,targets=n,valid=ok,status=meta.get('status','missing'),diagnostic_rows=len(diagnostics),projection_failures=sum(r['projection_converged'].lower()!='true' for r in diagnostics),max_noise_ratio=max((float(r['noise_ratio']) for r in diagnostics),default=None))
  for key in ['fit_seconds','separate_initial_cv_seconds','separate_initial_refit_seconds']:
   if key in meta:f[key]=float(meta[key])
  for key in ['process_seconds','cpu_seconds','max_rss_kib','process_exit']:
   if key in resource:f[key]=float(resource[key])
  fits.append(f)
  prim={int(r['target']):r for r in csv.DictReader((Path('outputs/gaussian-native')/c/f'{rep}.ours_warm.csv').open())}
  for r in rr:
   j=int(r['target']);r.update(case=c,rep=rep,primary_target=j in prim);rows.append(r)
   if j in prim:
    maxdiff=max(maxdiff,max(abs(float(r[k])-float(prim[j][k])) for k in ['estimate','se','lower','upper','pvalue']));npaired+=1
  if ok:
   for scope in ['primary10','all_slopes']:
    for role in ['active','null']:
     v=[r for r in rr if (scope=='all_slopes' or int(r['target']) in prim) and (float(r['truth'])!=0)==(role=='active')]
     clusters.append(dict(case=c,rep=rep,scope=scope,role=role,coverage=statistics.mean(float(r['lower'])<=float(r['truth'])<=float(r['upper']) for r in v),rejection=statistics.mean(float(r['pvalue'])<.05 for r in v),length=statistics.mean(float(r['upper'])-float(r['lower']) for r in v)))
D=Path('reports/gaussian-all-target-time20');D.mkdir(parents=True,exist_ok=True)
summary=[];timing=[]
for c in cases:
 ff=[f for f in fits if f['case']==c and f['valid']];t=dict(case=c,valid_datasets=len(ff),planned=20,targets=100 if c=='gaussian-benign' else 500)
 for key in ['fit_seconds','process_seconds','cpu_seconds','separate_initial_cv_seconds','separate_initial_refit_seconds']:
  v=[f[key] for f in ff if key in f]
  t[key+'_mean']=statistics.mean(v) if v else None;t[key+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None;t[key+'_median']=statistics.median(v) if v else None
 t['max_worker_rss_gib']=max((f['max_rss_kib']/1024**2 for f in ff if 'max_rss_kib' in f),default=None);timing.append(t)
 for scope in ['primary10','all_slopes']:
  for role in ['active','null']:
   cc=[r for r in clusters if r['case']==c and r['scope']==scope and r['role']==role];s=dict(case=c,scope=scope,role=role,valid=len(cc),planned=20)
   for k in ['coverage','rejection','length']:
    v=[r[k] for r in cc];s[k]=statistics.mean(v) if v else None;s[k+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
   summary.append(s)
checks=dict(planned_datasets=60,valid_datasets=sum(f['valid'] for f in fits),expected_target_rows=expected,target_rows=len(rows),unique_target_rows=len({(r['case'],r['rep'],r['target']) for r in rows}),errors=len(errors),primary_target_pairs=npaired,max_primary_target_list_difference=maxdiff,projection_failures=sum(f['projection_failures'] for f in fits),max_noise_ratio=max((f['max_noise_ratio'] for f in fits if f['max_noise_ratio'] is not None),default=None))
for name,obj in [('checks',checks),('timing',timing),('summary',summary),('fits',fits),('errors',errors)]:(D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
with (D/'records.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Automatic Gaussian all-slope timing on20matched datasets per design','','EMPIRICAL. The frozen production HC3/RLDPE policy now infers every slope:500moderate,500correlated,100benign. This matches hdi/EconML/sslasso target counts. Their native statistical recipes still differ. SIHR/hdm remain ten-target pipelines. Fixed-penalty matched-engine measurements are a separate experiment.','','|Design|Targets|Valid/planned|Warmed automatic fit mean ± MCSE (s)|Separate initial CV mean (s)|Process median (s)|Max worker RSS (GiB)|','|---|---:|---:|---:|---:|---:|---:|']
fmt=lambda v:'NA' if v is None else f'{v:.3f}'
for r in timing:lines.append(f"|{r['case']}|{r['targets']}|{r['valid_datasets']}/20|{fmt(r['fit_seconds_mean'])} ± {fmt(r['fit_seconds_mcse'])}|{fmt(r['separate_initial_cv_seconds_mean'])}|{fmt(r['process_seconds_median'])}|{fmt(r['max_worker_rss_gib'])}|")
lines+=['','```json',json.dumps(checks,indent=2),'```','','The process timer includes imports, small independent warmup, full automatic fit and a separate CV/initializer audit. It is not a pure single-fit cold-start measurement. Fit timers include each policy own tuning. Memory is a per-worker child-process peak. Statistical summaries are exploratory20-dataset checks, with dataset-cluster MCSE;22,000coefficients do not mean22,000independent replications. Primary ten-target estimates are compared with the saved ten-target calls; random coordinate order can change numerical iterates when target order changes. No policy was retuned. Zero empirical MCSE at a boundary does not imply certainty.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(checks))
