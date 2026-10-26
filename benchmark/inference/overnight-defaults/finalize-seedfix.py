import csv,json,math,re,statistics,sys
from pathlib import Path
stage,maxrep,prefix,job=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
D=Path('reports')/(stage+'-gaussian');rows=list(csv.DictReader((D/'records.csv').open()));di=list(csv.DictReader((D/'diagnostics.csv').open()))
assert len(rows)==len(di)==10*maxrep*10
by={};mp=mi=0.;hashes={}
for r in rows:
 assert r['converged'].lower()=='true'
 b,s=float(r['estimate']),float(r['se']);assert s>0
 mp=max(mp,abs(float(r['pvalue'])-math.erfc(abs(b/s)/math.sqrt(2))))
 mi=max(mi,abs(float(r['lower'])-(b-1.959963984540054*s)),abs(float(r['upper'])-(b+1.959963984540054*s)))
 by.setdefault((r['case'],int(r['rep'])),[]).append(r)
assert len(by)==10*maxrep and all(len(v)==10 for v in by.values()) and mp<1e-12 and mi<1e-12
for r in di:
 key=(r['case'],int(r['rep']));h=(r['input_x_sha256'],r['input_y_sha256'])
 assert all(re.fullmatch('[0-9a-f]{64}',v) for v in h) and (key not in hashes or hashes[key]==h)
 hashes[key]=h;assert float(r['noise_ratio'])<=1.5*(1+1e-8)
assert len(set(hashes.values()))==10*maxrep
rr=[]
for p in sorted(Path('resources').glob(prefix+'-*.txt')):
 r=dict(l.split('=',1) for l in p.read_text().splitlines() if '=' in l)
 assert all(k in r for k in ['cpu_seconds','process_seconds','max_rss_kib','process_exit']) and float(r['process_exit'])==0
 rr.append({k:float(v) for k,v in r.items()})
assert len(rr)==160
resource=dict(workers=len(rr),cpu_seconds=sum(r['cpu_seconds'] for r in rr),process_seconds_sum=sum(r['process_seconds'] for r in rr),max_worker_rss_gib=max(r['max_rss_kib'] for r in rr)/1024**2,job=job,note='64CPUs/128GiB over8nodes;160workers with16shards/design. Per-worker RSS,not simultaneous total. Earlier13-second allocation replacement separately retained in accounting.')
fits=[]
for c in sorted({r['case'] for r in rows}):
 for phase in ['first_per_worker','warm']:
  v=[float(v[0]['seconds']) for (case,rep),v in by.items() if case==c and ((rep<=16)==(phase=='first_per_worker'))]
  fits.append(dict(case=c,phase=phase,datasets=len(v),mean_seconds=statistics.mean(v),mcse_seconds=statistics.stdev(v)/math.sqrt(len(v)),median_seconds=statistics.median(v)))
summary=json.loads((D/'strict-summary.json').read_text());roles=json.loads((D/'per-slot.json').read_text());gates=json.loads((D/'calibration-gates.json').read_text())
flags=[dict(r,coverage_lower95_mc=r['coverage']-1.959963984540054*r['coverage_mcse'],null_upper95_mc=r['rejection']+1.959963984540054*r['rejection_mcse']) for r in roles if (float(r['truth'])!=0 and r['coverage']-1.959963984540054*r['coverage_mcse']<.925) or (float(r['truth'])==0 and r['rejection']+1.959963984540054*r['rejection_mcse']>.075)]
for name,obj in [('resource-summary',resource),('fit-time-summary',fits),('role-review',dict(unadjusted_role_flags=flags)),('additional-checks',dict(datasets=len(by),rows=len(rows),max_normal_pvalue_error=mp,max_interval_error=mi,distinct_hash_pairs=len(set(hashes.values()))))]:(D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
lines=['# Disjoint Gaussian validation after target-stream correction','','EMPIRICAL. Frozen statistical tuning policy, canonical physical-target random streams, ten prespecified Gaussian designs, '+str(maxrep)+' datasets/design. Base1000000000 is disjoint from the audited historical seed blocks. Source/protocol3eefe3b was pushed before generation. No tuning against this validation.','','|Design|Active coverage ± dataset MCSE (%)|Type I ± MCSE (%)|Power ± MCSE (%)|Active interval length ± MCSE|Warm10-target fit (s)|','|---|---:|---:|---:|---:|---:|']
for c in sorted({r['case'] for r in summary}):
 a=next(r for r in summary if r['case']==c and r['target_class']=='active');n=next(r for r in summary if r['case']==c and r['target_class']=='null');t=next(r for r in fits if r['case']==c and r['phase']=='warm')
 lines.append(f"|{c}|{100*a['coverage']:.2f} ± {100*a['coverage_mcse']:.2f}|{100*n['rejection']:.2f} ± {100*n['rejection_mcse']:.2f}|{100*a['rejection']:.2f} ± {100*a['rejection_mcse']:.2f}|{a['length']:.4f} ± {a['length_mcse']:.4f}|{t['mean_seconds']:.3f}|")
lines+=['',f"Pooled engineering gates passed:{sum(g['engineering_calibration_gate'] for g in gates)}/10. Unadjusted role flags:{len(flags)}. See per-slot.json and role-review.json. Tolerances are active lower95%MC bound>=92.5%,null upper95%MC bound<=7.5%,>=99%complete valid fits. These are not proof of uniform95%coverage. Correlated and weak-signal power and interval length remain part of the assessment.",'',f"All{len(by)}datasets and{len(rows)}target records are finite,converged and unique; zero numerical pvalue/interval discrepancies beyond1e-12. Consumed workerCPU:{resource['cpu_seconds']/3600:.2f}hours; maximum individual workerRSS:{resource['max_worker_rss_gib']:.3f}GiB. See accounting for job{job}. Fit timing includes initial CV and projections; first fit compilation and full process startup are separately scoped in fit-time-summary.json and resource receipts. These ten-target times are not all-slope competitor speedups.",'','Earlier seed-overlap and target-stream defects remain documented with their saved outputs. Historical numerical results were not deleted. The new confirmation addresses disjoint data streams and canonical target requests without changing tuning. Logistic/QP inference remains experimental. Marginal normal intervals only; joint/bootstrap calibration was not evaluated.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(dict(gates_passed=sum(g['engineering_calibration_gate'] for g in gates),role_flags=len(flags),resources=resource)))

bridge=[]
for p in Path('outputs/target-stream-bridge').glob('*/*.csv'):bridge.extend(csv.DictReader(p.open()))
assert len(bridge)==60 and sum(int(r['targets']) for r in bridge)==600
bs={k:max(float(r[k]) for r in bridge) for k in ['max_beta','max_se','max_pvalue','max_lambda']}
assert max(bs.values())<=1e-10
(D/'target-stream-bridge.json').write_text(json.dumps(dict(datasets=60,targets=600,max_differences=bs),indent=2)+'\n')
