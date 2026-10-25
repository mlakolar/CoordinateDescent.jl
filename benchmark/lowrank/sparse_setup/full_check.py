"""Independent actual-factor certificates; execute under Slurm."""
import csv, hashlib, json, pathlib, sys, tomllib
import numpy as np
from scipy import linalg as la
from common import CASES, load, read, EPS
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];base=root/'results'/phase
reps=range(1,21) if phase=='main' else [0]
records=[];errors=[];pairs=[];metadata=[]
def relative(A,B):return float(la.norm(A-B)/max(1.,la.norm(B)))
def rank(A):
    s=la.svdvals(A)
    return int((s>min(A.shape)*EPS*s[0]).sum()) if s.size else 0
variants=['public-baseline','public-candidate','common-baseline','common-candidate','r-public','r-common']
for case in CASES:
  for rep in reps:
    folder=root/'inputs'/f'{case}-{rep}';m,X,Y=load(folder);n,p,q=m['n'],m['p'],m['q']
    Xc=X-X.mean(0);Yc=Y-Y.mean(0);prepared=tomllib.loads((folder/'prepared.toml').read_text());states={}
    for variant in variants:
      dest=base/f'{case}-{rep}'/variant;isr=variant.startswith('r-')
      try:
        md=next(csv.DictReader((dest/'metadata.csv').open())) if isr else tomllib.loads((dest/'metadata.toml').read_text())
        ts=list(csv.DictReader((dest/'timings.csv').open()))
        if isr:
          assert md['X_md5']==m['X_md5'] and md['Y_md5']==m['Y_md5']
          probes={r['config']:r for r in csv.DictReader((dest/'probes.csv').open())}
          assert md['rrpack']=='0.1.14'
        else:
          assert md['X_sha256']==m['X_sha256'] and md['Y_sha256']==m['Y_sha256']
          assert md['julia_threads']==md['blas_threads']==1
        assert json.loads((dest/'resource.json').read_text())['exit_code']==0
        metadata.append(dict(case=case,rep=rep,variant=variant,**{k:v for k,v in md.items() if k not in ('case','rep','variant')}))
        configs=[(s,f,'unit') for s in (['r'] if isr else ['cd','fista']) for f in ['0.5','0.1','0.02']]
        if case=='medium':configs += [(s,'0.1','adaptive') for s in (['r'] if isr else ['cd','fista'])]
        assert len(ts)==3*len(configs)
        state={}
        for solver,frac,mode in configs:
          tag='-'.join([solver,frac,mode]);rounds=[r for r in ts if r['config']==tag]
          assert len(rounds)==3 and sorted(int(r['round']) for r in rounds)==[1,2,3]
          digest_key='B_md5' if isr else 'B_sha256';algorithm=hashlib.md5 if isr else hashlib.sha256
          assert len(set(r[digest_key] for r in rounds))==1
          assert algorithm((dest/f'{tag}-B.bin').read_bytes()).hexdigest()==rounds[0][digest_key]
          lam=float(frac)*prepared['lambda_adaptive' if mode=='adaptive' else 'lambda_unit']
          A=read(dest/f'{tag}-A.bin',(p,3));V=read(dest/f'{tag}-V.bin',(q,3));B=read(dest/f'{tag}-B.bin',(p,q));w=read(dest/f'{tag}-weights.bin',(p,))
          expected=read(folder/'adaptive.bin',(p,)) if mode=='adaptive' else np.ones(p)
          assert relative(w,expected)<=1e-12
          assert all(np.isfinite(z).all() for z in (A,V,B,w))
          E=Xc@A@V.T-Yc;G=Xc.T@E@V/n;an=la.norm(A,axis=1)
          kkt=np.maximum(0,la.norm(G,axis=1)-lam*w);nz=an>0
          kkt[nz]=la.norm(G[nz]+(lam*w[nz]/an[nz])[:,None]*A[nz],axis=1)
          kkt=float(kkt.max()/max(1.,la.norm(Xc.T@Yc@V/n)))
          GV=E.T@Xc@A/n;tangent=GV-V@((V.T@GV+GV.T@V)/2)
          response=float(la.norm(tangent)/max(1.,la.norm(Yc.T@Xc@A/n)))
          orthogonal=float(la.norm(V.T@V-np.eye(3)));reconstruction=relative(A@V.T,B)
          objective=float(np.sum(E**2)/(2*n)+lam*np.sum(w*an))
          rk=rank(B);passed=(kkt<=1e-6 and response<=1e-6 and orthogonal<=1e-10 and reconstruction<=1e-10 and rk<=3)
          if not isr:
            actual=tomllib.loads((dest/f'{tag}.toml').read_text());probe=tomllib.loads((dest/f'{tag}-probe.toml').read_text())
            returned=read(dest/f'{tag}-E.bin',(n,q));intercept=read(dest/f'{tag}-intercept.bin',(q,))
            support=np.flatnonzero(np.any(B!=0,axis=1));xr=rank(Xc[:,support]) if len(support) else 0
            df=rk*(xr+q-rk)+q
            passed=passed and abs(objective-actual['objective'])/max(1.,abs(objective))<=1e-10 and relative(returned,Yc-Xc@B)<=1e-10 and relative(intercept,Y.mean(0)-B.T@X.mean(0))<=1e-10 and actual['rank']==rk and actual['df']==df and actual['support']==(support+1).tolist() and actual['converged'] and all(r['converged']=='true' for r in rounds)
          else:
            probe=probes[tag];passed=passed and probe['converged']=='TRUE' and float(probe['replay_coefficient_error'])<=1e-12
          records.append(dict(case=case,rep=rep,variant=variant,config=tag,kkt=kkt,response=response,orthogonality=orthogonal,reconstruction=reconstruction,rank=rk,objective=objective,passed=bool(passed),probe=probe))
          state[tag]=(B,objective)
        states[variant]=state
      except Exception as e:errors.append(dict(case=case,rep=rep,variant=variant,error=repr(e)))
    for mode in ['public','common']:
      if all(v in states for v in [mode+'-baseline',mode+'-candidate','r-'+mode]):
        for tag,(B,objective) in states[mode+'-candidate'].items():
          for other,otag in [(mode+'-baseline',tag),('r-'+mode,'r-'+tag.split('-',1)[1])]:
            C,co=states[other][otag];gap=abs(co-objective)/max(1.,abs(objective),abs(co))
            pairs.append(dict(case=case,rep=rep,mode=mode,config=tag,comparator=other,objective_relative_difference=gap,coefficient_relative_difference=relative(C,B),matched_objective=gap<=1e-6))
expected=190*len(reps)
result=dict(records=len(records),expected_records=expected,failed_certificates=sum(not r['passed'] for r in records),errors=errors,certificates=records,pairs=pairs,metadata=metadata)
(base/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('certificates','pairs','metadata')},indent=2))
assert len(records)==expected and result['failed_certificates']==0 and not errors
