import csv,json,math,statistics
from pathlib import Path
D=Path('reports/liu-julia-confirmation500');D.mkdir(exist_ok=True,parents=True);root=Path('outputs/liu-julia-confirmation500');rows=[];missing=[];errors=[];summary=[]
for rep in range(1,501):
 if not (root/(str(rep)+'.complete')).exists():missing.append(rep)
 p=root/(str(rep)+'.csv')
 if p.exists():rows.extend(dict(r,rep=rep) for r in csv.DictReader(p.open()))
for p in root.glob('*.error'):errors.append(dict(file=p.name,error=p.read_text()))
for method in ['julia_logistic_default','julia_logistic_restricted_expert']:
 complete=[rep for rep in range(1,501) if len([r for r in rows if r['rep']==rep and r['method']==method])==2]
 for role in ['active','null']:
  rr=[r for r in rows if r['rep'] in complete and r['method']==method and (float(r['truth'])!=0)==(role=='active')];s=dict(method=method,role=role,valid=len(complete),planned=500)
  for k,fun in [('coverage',lambda r:float(r['lower'])<=float(r['truth'])<=float(r['upper'])),('rejection',lambda r:float(r['pvalue'])<.05),('length',lambda r:float(r['upper'])-float(r['lower'])),('seconds',lambda r:float(r['seconds']))]:
   v=[float(fun(r)) for r in rr];s[k]=statistics.mean(v) if v else None;s[k+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
  summary.append(s)
rr=[]
for p in Path('resources').glob('liu-julia-confirmation500-*.txt'):
 r=dict(l.split('=',1) for l in p.read_text().splitlines() if '=' in l);rr.append({k:float(v) for k,v in r.items()})
resource=dict(workers=len(rr),cpu_seconds=sum(r['cpu_seconds'] for r in rr),max_worker_rss_gib=max(r['max_rss_kib'] for r in rr)/1024**2)
for name,obj in [('summary',summary),('checks',dict(planned_target_records=2000,records=len(rows),unique_records=len({(r['rep'],r['method'],r['target']) for r in rows}),errors=len(errors),missing_markers=missing)),('errors',errors),('resource-summary',resource)]: (D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
with (D/'records.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Julia logistic recipes on the Liu positive-control development data','','EMPIRICAL.500matched independent confirmation datasets,n1000,p201,first active and last null role. Native automatic CV recipe and previously frozen restricted-LDPE expert recipe; no new tuning. Same input files as official DoubleML development100. No production logistic default is changed.','','|Method|Role|Complete/planned|Coverage ± MCSE (%)|Rejection ± MCSE (%)|Length|','|---|---|---:|---:|---:|---:|']
for s in summary:lines.append(f"|{s['method']}|{s['role']}|{s['valid']}/500|{100*s['coverage']:.2f} ±{100*s['coverage_mcse']:.2f}|{100*s['rejection']:.2f} ±{100*s['rejection_mcse']:.2f}|{s['length']:.4f}|")
(D/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
