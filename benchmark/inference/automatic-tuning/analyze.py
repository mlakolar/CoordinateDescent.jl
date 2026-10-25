"""Aggregate saved Discovery outputs; do not refit."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
root=Path.cwd();groups=defaultdict(list);coeff=defaultdict(list)
for f in Path('study').glob('*.csv'):
    stage,casefamily,caselabel,shard=f.stem.split('-');case=casefamily+'-'+caselabel
    for r in csv.DictReader(f.open()):groups[stage,case,r['method']].append(r)
    cf=Path(str(f)+'.coefficients')
    if cf.exists():
        for row in csv.reader(cf.open()):coeff[stage,case,row[1]].append([int(row[0]),int(row[2]),*map(float,row[3:])])
fields=['coverage_active','coverage_null','type1','power','length_active','length_null','sigma','lambda','node_lambda','fit_seconds','signed_bias_active','signed_z_bias_active']
rows=[]
for (stage,case,method),rs in sorted(groups.items()):
    d=dict(stage=stage,case=case,method=method,datasets=len(rs),converged=sum(r['converged']=='true' for r in rs))
    for name in fields:
        a=np.array([float(r[name]) for r in rs]);d[name]=float(a.mean());d[name+'_mcse']=float(a.std(ddof=1)/np.sqrt(len(a)));d[name+'_median']=float(np.median(a))
    cc=np.array(coeff[stage,case,method]);a=cc[:,2]!=0
    # Per-target variability across replications, then average variance ratio.
    ratios=[]
    for j in np.unique(cc[a,1]):
        z=cc[cc[:,1]==j];ratios.append(float(z[:,3].std(ddof=1)/np.sqrt(np.mean(z[:,4]**2))))
    d['empirical_sd_over_rms_se_active']=float(np.mean(ratios));rows.append(d)
with open('study-summary.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def meta(path):
 return dict(line.rstrip().split('=',1) for line in path.read_text().splitlines() if '=' in line)
bench=[];parity=[]
for f in Path('bench').glob('*.meta'):
    engine,label,rep=f.stem.split('-');d=dict(engine=engine,case=label,rep=int(rep),**meta(f))
    t=Path(str(f).replace('.meta','.time'))
    if t.exists():d.update(meta(t))
    bench.append(d)
for label in ['moderate','correlated']:
 for rep in range(1,11):
    common=dict(case=label,rep=rep)
    files=[Path('bench')/f'{e}-{label}-{rep}-coef.csv' for e in ['ours','hdi','econml']]
    if all(f.exists() for f in files):
     a,b,c=[np.loadtxt(f,delimiter=',') for f in files]
     parity.append(dict(**common,comparison='ours-hdi',beta_max_abs=float(np.max(abs(a[:,0]-b[:,0]))),se_max_abs=float(np.max(abs(a[:,1]-b[:,1]))),se_max_relative=float(np.max(abs(a[:,1]/b[:,1]-1)))))
     parity.append(dict(**common,comparison='ours-econml',beta_max_abs=float(np.max(abs(a[:,0]-c[:,0]))),se_max_abs=float(np.max(abs(a[:,1]-c[:,1]))),se_max_relative=float(np.max(abs(a[:,1]/c[:,1]-1)))))
    files=[Path('bench')/f'{e}-{label}-{rep}-theta.csv' for e in ['oursqp','sslasso']]
    if all(f.exists() for f in files):
     a,b=[np.loadtxt(f,delimiter=',') for f in files];parity.append(dict(**common,comparison='oursqp-sslasso',theta_max_abs=float(np.max(abs(a-b)))))
summary=[]
for label in ['moderate','correlated']:
 for engine in ['ours','hdi','econml','oursqp','sslasso','SIHR','oursqp10']:
    rs=[r for r in bench if r['engine']==engine and r['case']==label]
    ok=[r for r in rs if r.get('status')=='ok' and r.get('process_exit')=='0']
    d=dict(case=label,engine=engine,expected=10,records=len(rs),valid=len(ok))
    if ok:
     for field in ['fit_seconds','process_seconds','max_rss_kib']:
      vals=[float(r[field]) for r in ok];d[field+'_median']=float(np.median(vals));d[field+'_max']=max(vals)
    summary.append(d)
validation=dict(expected_automatic_datasets=800,automatic_datasets=sum(r['datasets'] for r in rows if r['method']=='automatic'),automatic_converged=sum(r['converged'] for r in rows if r['method']=='automatic'),expected_benchmark_fits=140,benchmark_fits=len(bench),benchmark_valid=sum(r.get('status')=='ok' and r.get('process_exit')=='0' for r in bench),study_error_files=[{'path':str(f),'errors':f.read_text()} for f in Path('study').glob('*.errors')],parity=parity)
Path('validation.json').write_text(json.dumps(validation,indent=2)+'\n');Path('benchmark-summary.json').write_text(json.dumps(summary,indent=2)+'\n');Path('benchmark-records.json').write_text(json.dumps(bench,indent=2)+'\n')
lines=['# Automatic tuning follow-up','', 'Automatic policy frozen before development/holdout execution. Fresh holdout:100 datasets per case. Nominal marginal coverage95%, unadjusted typeI5%. Errors/uncertainty remain in machine-readable files.','', '| Holdout case | Active coverage | Null coverage | Type I | Power | Length active/null | Median automatic seconds |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['stage']=='holdout' and r['method']=='automatic':
  lines.append(f"| {r['case']} | {r['coverage_active']:.3f} ± {r['coverage_active_mcse']:.3f} | {r['coverage_null']:.3f} ± {r['coverage_null_mcse']:.3f} | {r['type1']:.3f} | {r['power']:.3f} | {r['length_active']:.3f}/{r['length_null']:.3f} | {r['fit_seconds_median']:.3f} |")
lines+=['','± denotes one dataset-level Monte Carlo standard error, not a confidence limit. Targets are five active and five boundary-null coefficients.','', '| Speed case | Engine | Valid/expected | Median fit seconds | Median process seconds | Max RSS MiB |','|---|---|---:|---:|---:|---:|']
for r in summary:
 if r['valid']:
  lines.append(f"| {r['case']} | {r['engine']} | {r['valid']}/10 | {r['fit_seconds_median']:.4f} | {r['process_seconds_median']:.3f} | {r['max_rss_kib_max']/1024:.1f} |")
 else:lines.append(f"| {r['case']} | {r['engine']} | 0/10 | — | — | — |")
lines+=['','Nodewise engines:500 slopes, fixed common penalties, residual-DF noise estimation. QP engines:500 projection rows at common mu, with numerical certificates. SIHR and oursqp10:10-target complete pipelines with different inferential procedures; SIHR retains default variance inflation and filtering. Tuning time is outside fixed-penalty speed timings.','',f"Validation: {validation['automatic_converged']}/800 automatic fits converged; {validation['benchmark_valid']}/140 benchmark fits valid. Full1000-replication run not submitted."]
Path('RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in validation.items() if k not in ['parity','study_error_files']},indent=2))
