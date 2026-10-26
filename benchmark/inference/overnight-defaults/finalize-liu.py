import collections,csv,json,math,sys
from pathlib import Path
stage,n,prefix,job=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
D=Path('reports')/('liu-doubleml-confirmation500' if stage=='liu-confirmation' else 'liu-doubleml-development100')
rr=[]
for rep in range(1,n+1):
 for score in ['instrument','nuisance_space']:
  if stage=='development' and rep<=20:p=Path('resources')/f'doubleml-sparse-development-liu-published-{rep}-{score}.txt'
  else:p=Path('resources')/f'{prefix}-{rep}-{score}.txt'
  if p.exists():
   d=dict(l.split('=',1) for l in p.read_text().splitlines() if '=' in l);assert float(d['process_exit'])==0;rr.append({k:float(v) for k,v in d.items()})
assert len(rr)==2*n
resource=dict(workers=len(rr),cpu_seconds=sum(r['cpu_seconds'] for r in rr),process_seconds_sum=sum(r['process_seconds'] for r in rr),max_worker_rss_gib=max(r['max_rss_kib'] for r in rr)/1024**2,job=job,note='Fresh worker process per dataset/score, two target roles. Per-worker peakRSS,not simultaneous total. Development reuses first20original receipts; native tuning/cross-fitting time included.')
records=list(csv.DictReader((D/'records.csv').open()));di=[];warn=collections.Counter()
for rep in range(1,n+1):
 for score in ['instrument','nuisance_space']:
  stem=Path('outputs')/stage/'liu-published'/(str(rep)+'.doubleml_sparse_'+score)
  p=Path(str(stem)+'.diagnostics.csv')
  if p.exists():di.extend(csv.DictReader(p.open()))
  p=Path(str(stem)+'.warnings.json')
  if p.exists():warn.update(r['category'] for r in json.loads(p.read_text()))
keys={(r['rep'],r['method'],r['target']) for r in records};assert len(records)==len(keys) and len(di)==len(records)
mp=max(abs(float(r['pvalue'])-math.erfc(abs(float(r['estimate'])/float(r['se']))/math.sqrt(2))) for r in records)
sp=max(float(r['se_parity']) for r in di);sr=max(abs(float(r['score_mean'])) for r in di)
assert mp<1e-10 and sp<1e-8 and sr<1e-6
for name,obj in [('resource-summary',resource),('additional-checks',dict(unique_finite_target_records=len(records),normal_pvalue_max_error=mp,score_sandwich_se_max_error=sp,score_mean_max_residual=sr,warning_categories=dict(warn)))]:
 (D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
s=(D/'RESULTS.md').read_text();s+=f"\nNative numerical checks: SEparity{sp:.3g}, maximum normal-pvalue error{mp:.3g}, score residual{sr:.3g}. Actual workerCPU{resource['cpu_seconds']/3600:.3f}hours, maximum individual-workerRSS{resource['max_worker_rss_gib']:.3f}GiB. Warning categories:{dict(warn)}. See resource-summary.json and exact-job accounting.\n"
if (D/'prospective-gate.json').exists():s+='\nProspective primary instrument gate:\n\n```json\n'+(D/'prospective-gate.json').read_text()+'```\n'
(D/'RESULTS.md').write_text(s);print(json.dumps(resource))
