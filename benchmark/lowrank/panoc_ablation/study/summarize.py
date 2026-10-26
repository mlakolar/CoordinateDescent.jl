import sys,pathlib,json,tomllib,csv,statistics,hashlib,math
from config import CASES,PRIMARY,CONTROLS
root=pathlib.Path(sys.argv[1]);canary='--canary' in sys.argv;processes=[];failures=[];certs=[];rows=[];pathlayout='*/*/resource.json' if canary else '*/rep*/*/resource.json'
for resourcefile in sorted(root.glob(pathlayout)):
 f=resourcefile.parent;method=f.name;case=f.parent.name if canary else f.parent.parent.name;rep=0 if canary else int(f.parent.name[3:]);resource=json.loads(resourcefile.read_text());folder=pathlib.Path('inputs/pilot-v1')/case if canary else pathlib.Path('inputs/study-v1')/case/('rep'+str(rep));base=dict(case=case,rep=rep,method=method,resource=resource)
 try:
  md=tomllib.loads((f/'metadata.toml').read_text()) if (f/'metadata.toml').exists() else json.loads((f/'metadata.json').read_text());base['metadata']=md
  assert all(md['input_hashes'][k]==hashlib.sha256((folder/(k+'.bin')).read_bytes()).hexdigest() for k in ['X','Y'])
  assert 'EPYC 7513' in md['cpu']
  if method.startswith('py_'):assert all(r['num_threads']==1 for r in md['threadpools'])
  else:assert md['blas_threads']==md['julia_threads']==1
  cs=json.loads((f/'certificates.json').read_text());certs.extend(dict(case=case,rep=rep,method=method,**r) for r in cs)
  ts=list(csv.DictReader((f/'timings.csv').open()));workloads={}
  for r in ts:workloads.setdefault((r['mode'],r['kind'],r['fraction']),[]).append(r)
  for (mode,kind,fraction),rs in workloads.items():
   assert len(rs)==3 and len({r['coefficient_sha256'] for r in rs})==1
   matched=[r for r in cs if r['mode']==mode and r['kind']==kind and (kind=='path' or abs(r['fraction']-float(fraction))<1e-14)]
   passed=bool(matched) and all(r['passed'] for r in matched) and (kind!='path' or len(matched)==8)
   rows.append(dict(case=case,rep=rep,method=method,mode=mode,kind=kind,fraction=float(fraction),seconds=statistics.median(float(r['seconds']) for r in rs),allocated=None if not rs[0]['bytes'] else statistics.median(int(r['bytes']) for r in rs),all_accuracy_pass=passed,iterations=int(rs[0]['iterations'])))
  if resource['exit_code'] or md['failures'] or not all(r['passed'] for r in cs):raise ValueError(dict(exit_code=resource['exit_code'],failures=md['failures'],failed_certificates=sum(not r['passed'] for r in cs)))
  assert len(cs)==22 and len(workloads)==8
  if not method.startswith('py_') and method!='apg':
   counts=tomllib.loads((f/'counts.toml').read_text())['records'];assert len(counts)==8 and all(r['matched_timed_B_and_iterations'] for r in counts)
  base['timings']=ts;processes.append(base)
 except Exception as e:failures.append(dict(**base,error=repr(e)))
expected=30 if canary else 1800
report=dict(expected_processes=expected,successful_processes=len(processes),failed_processes=len(failures),processes=processes,failures=failures,certificates=certs,medians=rows)
(root/'summary.json').write_text(json.dumps(report,indent=2))
for name,data in [('medians',rows),('certificates',certs)]:
 if data:
  with (root/(name+'.csv')).open('w') as f:
   fields=list(dict.fromkeys(k for r in data for k in r));w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(data)
print(json.dumps(dict(expected=expected,successful=len(processes),failures=[{k:v for k,v in r.items() if k not in ['metadata']} for r in failures],certificates=len(certs),failed_certificates=sum(not r['passed'] for r in certs)),indent=2))
if canary:assert len(processes)==30 and not failures and len(certs)==660
else:assert len(processes)+len(failures)==1800
