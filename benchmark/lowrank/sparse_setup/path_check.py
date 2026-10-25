import csv,hashlib,json,pathlib,sys,tomllib
import numpy as np
from scipy import linalg as la
from common import CASES,load,read,EPS
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];base=root/'results'/phase
reps=range(1,21) if phase=='main' else [0]
records=[];errors=[];pairs=[];rows=[];components=[]
def relative(A,B):return float(la.norm(A-B)/max(1.,la.norm(B)))
def rank(A):
  s=la.svdvals(A)
  return int((s>min(A.shape)*EPS*s[0]).sum()) if len(s) else 0
for case in CASES:
 for rep in reps:
  folder=root/'inputs'/f'{case}-{rep}';m,X,Y=load(folder);n,p,q=m['n'],m['p'],m['q'];Xc=X-X.mean(0);Yc=Y-Y.mean(0)
  states={}
  for variant in ['baseline','candidate']:
   dest=base/f'{case}-{rep}'/('path-'+variant)
   try:
    assert json.loads((dest/'resource.json').read_text())['exit_code']==0
    md=tomllib.loads((dest/'metadata.toml').read_text());assert md['julia_threads']==md['blas_threads']==1
    ts=list(csv.DictReader((dest/'timings.csv').open()));rows.extend(ts)
    cs=list(csv.DictReader((dest/'components.csv').open()));components.extend(cs);assert len(cs)==90
    configs=[s+'-'+mode for s in ['cd','fista'] for mode in (['unit','adaptive'] if case=='medium' else ['unit'])]
    assert len(ts)==3*len(configs);state={}
    for tag in configs:
     rounds=[r for r in ts if r['config']==tag];assert len(rounds)==3 and len(set(r['B_sha256'] for r in rounds))==1
     digest=hashlib.sha256()
     for j in range(1,9):digest.update((dest/f'{tag}-{j}-B.bin').read_bytes())
     assert digest.hexdigest()==rounds[0]['B_sha256']
     assert all(r['converged']=='true' for r in rounds)
     for j in range(1,9):
      label=f'{tag}-{j}';actual=tomllib.loads((dest/f'{label}.toml').read_text());lam=actual['lambda']
      A=read(dest/f'{label}-A.bin',(p,3));V=read(dest/f'{label}-V.bin',(q,3));B=read(dest/f'{label}-B.bin',(p,q));w=read(dest/f'{label}-weights.bin',(p,))
      expected=read(folder/'adaptive.bin',(p,)) if tag.endswith('adaptive') else np.ones(p);assert relative(w,expected)<=1e-12
      E=Xc@A@V.T-Yc;G=Xc.T@E@V/n;an=la.norm(A,axis=1)
      kkt=np.maximum(0,la.norm(G,axis=1)-lam*w);nz=an>0
      kkt[nz]=la.norm(G[nz]+(lam*w[nz]/an[nz])[:,None]*A[nz],axis=1)
      kkt=float(kkt.max()/max(1.,la.norm(Xc.T@Yc@V/n)))
      GV=E.T@Xc@A/n;tangent=GV-V@((V.T@GV+GV.T@V)/2)
      response=float(la.norm(tangent)/max(1.,la.norm(Yc.T@Xc@A/n)))
      obj=float(np.sum(E**2)/(2*n)+lam*np.sum(w*an));rk=rank(B);support=np.flatnonzero(np.any(B!=0,axis=1));xr=rank(Xc[:,support]) if len(support) else 0
      ok=(actual['converged'] and kkt<=1e-6 and response<=1e-6 and la.norm(V.T@V-np.eye(3))<=1e-10 and relative(A@V.T,B)<=1e-10 and rk<=3 and
        actual['rank']==rk and actual['df']==rk*(xr+q-rk)+q and actual['support']==(support+1).tolist() and
        abs(obj-actual['objective'])/max(1.,abs(obj))<=1e-10 and relative(read(dest/f'{label}-E.bin',(n,q)),Yc-Xc@B)<=1e-10 and
        relative(read(dest/f'{label}-intercept.bin',(q,)),Y.mean(0)-B.T@X.mean(0))<=1e-10)
      records.append(dict(case=case,rep=rep,variant=variant,config=tag,level=j,kkt=kkt,response=response,objective=obj,passed=bool(ok)))
      state[label]=(B,obj)
    states[variant]=state
   except Exception as e:errors.append(dict(case=case,rep=rep,variant=variant,error=repr(e)))
  if len(states)==2:
   for label,(B,obj) in states['baseline'].items():
    C,co=states['candidate'][label];gap=abs(co-obj)/max(1.,abs(obj),abs(co))
    pairs.append(dict(case=case,rep=rep,config=label,objective_relative_difference=gap,coefficient_relative_difference=relative(C,B),matched_objective=gap<=1e-6))
result=dict(records=len(records),expected_records=224*len(reps),failed_certificates=sum(not r['passed'] for r in records),errors=errors,certificates=records,pairs=pairs)
(base/'path-validation.json').write_text(json.dumps(result,indent=2))
for name,items in [('path-timings.csv',rows),('components.csv',components)]:
 if items:
  with (base/name).open('w') as f:
   w=csv.DictWriter(f,list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)
print(json.dumps({k:v for k,v in result.items() if k not in ['certificates','pairs']},indent=2))
assert len(records)==result['expected_records'] and not errors and result['failed_certificates']==0
