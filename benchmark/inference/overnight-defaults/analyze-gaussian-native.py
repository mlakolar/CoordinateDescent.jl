"""Discovery saved-output analysis; explicit planned groups, native warnings/certificates."""
import csv,json,math,statistics,sys
from pathlib import Path
stage,maxrep=sys.argv[1],int(sys.argv[2]);cases=['gaussian-moderate','gaussian-correlated','gaussian-benign'];engines=['ours_cold','ours_warm','hdi','EconML','sslasso','SIHR','hdm']
D=Path('reports')/('gaussian-native-'+stage);D.mkdir(parents=True,exist_ok=True)
allrows=[];fits=[];summary=[];resources=[]
for c in cases:
 for e in engines:
  clusters={'active':[],'null':[]};valid=available=complete=0;group=[]
  for rep in range(1,maxrep+1):
   base=Path('outputs/gaussian-native')/c/f'{rep}.{e}'
   certbase=Path('outputs/gaussian-native-cert-v3')/c/f'{rep}.{e}'
   if e=='sslasso' and Path(str(certbase)+'.complete').exists():base=certbase
   meta={};warning='';record=dict(case=c,engine=e,rep=rep)
   if e=='EconML' and Path(str(base)+'.json').exists():
    meta=json.loads(Path(str(base)+'.json').read_text());warning=' | '.join(w['category']+': '+w['message'] for w in meta.get('warnings',[]))
   elif Path(str(base)+'.meta').exists():meta=dict(l.split('=',1) for l in Path(str(base)+'.meta').read_text().splitlines() if '=' in l);warning=meta.get('warnings','')
   raw=list(csv.DictReader(Path(str(base)+'.csv').open())) if Path(str(base)+'.csv').exists() else []
   if Path(str(base)+'.complete').exists():complete+=1
   ownerror=Path(str(base)+'.error');warnconv=any(s in warning.lower() for s in ['convergencewarning','did not converge','not converge','failed to converge','numerically 0 or 1'])
   finite=len(raw)==10 and len({int(r['target']) for r in raw})==10 and all(all(math.isfinite(float(r[k])) for k in ['estimate','lower','upper','pvalue']) and 0<=float(r['pvalue'])<=1 and float(r['upper'])>float(r['lower']) for r in raw)
   intervalavailable=meta.get('status')=='ok' and finite and not ownerror.exists()
   certified=True
   if e=='sslasso':certified=str(meta.get('all_qp_rows_certified','False')).lower()=='true'
   fitvalid=intervalavailable and not warnconv and certified
   available+=intervalavailable;valid+=fitvalid
   record.update(status=meta.get('status','missing'),complete=Path(str(base)+'.complete').exists(),interval_available=intervalavailable,convergence_warning=warnconv,qp_certified=certified,strict_valid=fitvalid,warnings=warning,error=ownerror.read_text() if ownerror.exists() else meta.get('error',''))
   for key in ['fit_seconds','actual_inferred_targets','separate_initial_cv_seconds','separate_initial_refit_seconds','qp_max_abs_gap']:
    if key in meta:
     try:record[key]=float(meta[key])
     except (ValueError,TypeError):record[key]=meta[key]
   fits.append(record);group.append(record)
   # Conditional raw-native coverage is retained separately from strict solve availability.
   if intervalavailable:
    for role in ['active','null']:
     rr=[r for r in raw if (float(r['truth'])!=0)==(role=='active')]
     vals=dict(coverage=statistics.mean(float(r['lower'])<=float(r['truth'])<=float(r['upper']) for r in rr),rejection=statistics.mean(float(r['pvalue'])<.05 for r in rr),length=statistics.mean(float(r['upper'])-float(r['lower']) for r in rr))
     clusters[role].append(vals)
   for r in raw:r.update(case=c,rep=rep,engine=e,strict_valid=fitvalid,interval_available=intervalavailable);allrows.append(r)
  for role,cs in clusters.items():
   s=dict(case=c,engine=e,role=role,planned=maxrep,complete=complete,interval_available=available,strict_valid=valid,coverage_scope='Conditional on finite native output, including convergence-warned/uncertified rows; strict availability shown separately. Not a validated default when strict availability fails.')
   for k in ['coverage','rejection','length']:
    v=[r[k] for r in cs];s[k]=statistics.mean(v) if v else None;s[k+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
   times=[r['fit_seconds'] for r in group if 'fit_seconds' in r];s['mean_fit_seconds']=statistics.mean(times) if times else None
   summary.append(s)
  for rep in range(1,maxrep+1):
   prefix='gaussian-native-canary' if rep==1 else 'gaussian-native-pilot20'
   rp=Path('resources')/f'{prefix}-{e}-{c}-{rep}.txt'
   if rep==1 and e=='sslasso':
    recovery=Path('resources')/f'gaussian-native-recovery-{e}-{c}-{rep}.txt'
    if recovery.exists():rp=recovery
   certresource=Path('resources')/f'gaussian-native-cert-v3-{c}-{rep}.txt'
   if e=='sslasso' and certresource.exists():rp=certresource
   if rp.exists():
    d=dict(l.split('=',1) for l in rp.read_text().splitlines() if '=' in l)
    resources.append(dict(case=c,engine=e,rep=rep,**{k:float(v) for k,v in d.items()}))
for name,obj in [('fits',fits),('summary',summary),('resource-records',resources)]:(D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
if allrows:
 with (D/'records.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
checks=dict(planned_fits=3*7*maxrep,complete_fits=sum(f['complete'] for f in fits),interval_available=sum(f['interval_available'] for f in fits),strict_valid=sum(f['strict_valid'] for f in fits),convergence_warning_fits=sum(f['convergence_warning'] for f in fits),resources=len(resources),scope='Native/default comparison; own10-target policy versus hdi/EconML/sslasso all slopes and SIHR/hdm ten requested targets. Not matched-penalty engine timing.')
(D/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
if resources:
 total=sum(r['process_seconds'] for r in resources);rss=max(r['max_rss_kib'] for r in resources)/1024**2
 forecast=dict(scope='Only these3Gaussian designs and7tasks including separatecold/warm own fits;not expanded10design forecast. Canary-only forecast is highly uncertain.',observed_process_seconds=total,max_worker_rss_gib=rss,pilot20_process_cpu_hours=total/maxrep*20/3600,three_case1000_process_cpu_hours=total/maxrep*1000/3600,three_case1000_ideal_wall_hours64cpu=total/maxrep*1000/3600/64,planning_margin=1.5,three_case1000_planning_wall_hours64cpu=total/maxrep*1000/3600/64*1.5)
 (D/'forecast.json').write_text(json.dumps(forecast,indent=2)+'\n')
print(json.dumps(checks))
if stage.startswith('canary'):
 assert checks['complete_fits']==21 and checks['resources']==21
 assert all(f['interval_available'] for f in fits if f['engine'].startswith('ours_'))
