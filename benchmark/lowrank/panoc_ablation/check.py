import pathlib,json,csv,tomllib,statistics,numpy as np
from scipy import linalg as la
root=pathlib.Path('results/pilot-v1');certs=[];tables=[];failures=[]
for out in sorted(root.iterdir()):
 if not out.is_dir():continue
 case=out.name;meta=json.loads((pathlib.Path('inputs/pilot-v1')/case/'meta.json').read_text());n,p,q=(meta[k] for k in ['n','p','q'])
 X=np.fromfile(pathlib.Path('inputs/pilot-v1')/case/'X.bin',dtype='<f8').reshape((n,p),order='F');Y=np.fromfile(pathlib.Path('inputs/pilot-v1')/case/'Y.bin',dtype='<f8').reshape((n,q),order='F')
 X-=X.mean(0);Y-=Y.mean(0);C=X.T@Y/n;L=meta['L'];scale=max(1.,la.norm(C))
 md=tomllib.loads((out/'metadata.toml').read_text());failures.extend(dict(case=case,**r) for r in md['failures'])
 for f in sorted(out.glob('*.bin')):
  name=f.stem
  if '-path-' in name:
   j=int(name.rsplit('-',1)[1])-1;lam=np.geomspace(.5,.02,8)[j]*meta['lambda_max']
  else:lam=.1*meta['lambda_max']
  B=np.fromfile(f,dtype='<f8').reshape((p,q),order='F');R=Y-X@B;G=-X.T@R/n;U,s,Vt=la.svd(B-G/L,full_matrices=False);prox=(U*np.maximum(s-lam/L,0))@Vt
  mapping=L*la.norm(B-prox)/scale;primal=np.sum(R*R)/(2*n)+lam*la.svdvals(B).sum();dualR=R/n*min(1.,lam/max(la.svdvals(G)[0],np.finfo(float).tiny));dual=np.sum(Y*dualR)-n*np.sum(dualR*dualR)/2;gap=max(0.,primal-dual)/max(1.,abs(primal),abs(dual))
  certs.append(dict(case=case,fit=name,objective=primal,gap=gap,mapping=mapping,passed=bool(gap<=1e-6 and mapping<=1e-6)))
 timing=list(csv.DictReader((out/'timings.csv').open()))
 for config in sorted({r['config'] for r in timing}):
  for mode in ['public','cached']:
   rows=[r for r in timing if r['config']==config and r['mode']==mode]
   if len(rows)==3:tables.append(dict(case=case,config=config,mode=mode,seconds=statistics.median(float(r['seconds']) for r in rows),bytes=statistics.median(int(r['bytes']) for r in rows)))
report=dict(phase='calibration_only',independent_inputs_per_case=1,table=tables,certificates=certs,failures=failures)
(root/'summary.json').write_text(json.dumps(report,indent=2))
for label,rows in [('timing-summary',tables),('certificates',certs)]:
 with (root/(label+'.csv')).open('w') as f:
  w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(dict(certificates=len(certs),certificate_failures=sum(not r['passed'] for r in certs),adapter_failures=failures),indent=2))
