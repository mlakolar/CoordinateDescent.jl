import csv,hashlib,json,pathlib,sys,tomllib
import numpy as np
from scipy import linalg as la
from common import CASES,load,read,EPS
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2] if len(sys.argv)>2 else 'pilot';base=root/'results'/phase;records=[];errors=[];pairs=[]
def relative(A,B):return float(la.norm(A-B)/max(1.,la.norm(B)))
def rank(A):
    s=la.svdvals(A)
    return int((s>min(A.shape)*EPS*s[0]).sum()) if s.size else 0
for case in CASES:
    folder=root/'inputs'/f'{case}-0';m,X,Y=load(folder);n,p,q=m['n'],m['p'],m['q']
    Xc=X-X.mean(0);Yc=Y-Y.mean(0);prepared=tomllib.loads((folder/'prepared.toml').read_text());states={}
    for variant in ('baseline','candidate'):
        dest=base/f'{case}-0'/variant
        try:
            md=tomllib.loads((dest/'metadata.toml').read_text());ts=list(csv.DictReader((dest/'timings.csv').open()))
            assert md['X_sha256']==m['X_sha256'] and md['Y_sha256']==m['Y_sha256']
            assert md['julia_threads']==md['blas_threads']==1
            assert json.loads((dest/'resource.json').read_text())['exit_code']==0
            state={}
            for tag in sorted(set(r['config'] for r in ts)):
                rounds=[r for r in ts if r['config']==tag]
                assert len(rounds)==3 and len(set(r['B_sha256'] for r in rounds))==1
                assert hashlib.sha256((dest/f'{tag}-B.bin').read_bytes()).hexdigest()==rounds[0]['B_sha256']
                solver,frac,mode=tag.split('-');lam=float(frac)*prepared['lambda_adaptive' if mode=='adaptive' else 'lambda_unit']
                A=read(dest/f'{tag}-A.bin',(p,3));V=read(dest/f'{tag}-V.bin',(q,3));B=read(dest/f'{tag}-B.bin',(p,q));w=read(dest/f'{tag}-weights.bin',(p,))
                expected=read(folder/'adaptive.bin',(p,)) if mode=='adaptive' else np.ones(p)
                assert relative(w,expected)<=1e-12
                E=Xc@A@V.T-Yc;G=Xc.T@E@V/n;an=la.norm(A,axis=1)
                kkt=np.maximum(0,la.norm(G,axis=1)-lam*w);nz=an>0
                kkt[nz]=la.norm(G[nz]+(lam*w[nz]/an[nz])[:,None]*A[nz],axis=1)
                kkt=float(kkt.max()/max(1.,la.norm(Xc.T@Yc@V/n)))
                GV=E.T@Xc@A/n;tangent=GV-V@((V.T@GV+GV.T@V)/2)
                response=float(la.norm(tangent)/max(1.,la.norm(Yc.T@Xc@A/n)))
                orthogonal=float(la.norm(V.T@V-np.eye(3)))
                reconstruction=relative(A@V.T,B)
                objective=float(np.sum(E**2)/(2*n)+lam*np.sum(w*an))
                actual=tomllib.loads((dest/f'{tag}.toml').read_text())
                returned=read(dest/f'{tag}-E.bin',(n,q));intercept=read(dest/f'{tag}-intercept.bin',(q,))
                support=np.flatnonzero(np.any(B!=0,axis=1));rk=rank(B);xr=rank(Xc[:,support]) if len(support) else 0
                df=rk*(xr+q-rk)+q
                passed=(kkt<=1e-6 and response<=1e-6 and orthogonal<=1e-10 and reconstruction<=1e-10 and rk<=3 and
                    abs(objective-actual['objective'])/max(1.,abs(objective))<=1e-10 and
                    relative(returned,Yc-Xc@B)<=1e-10 and relative(intercept,Y.mean(0)-B.T@X.mean(0))<=1e-10 and
                    actual['rank']==rk and actual['df']==df and actual['support']==(support+1).tolist() and actual['converged'] and all(r['converged']=='true' for r in rounds))
                records.append(dict(case=case,variant=variant,config=tag,kkt=kkt,response=response,orthogonality=orthogonal,reconstruction=reconstruction,objective=objective,passed=bool(passed)))
                state[tag]=(B,objective)
            states[variant]=state
        except Exception as e:errors.append(dict(case=case,variant=variant,error=repr(e)))
    if len(states)==2:
        for tag,(B,objective) in states['baseline'].items():
            C,co=states['candidate'][tag];gap=abs(co-objective)/max(1.,abs(objective))
            pairs.append(dict(case=case,config=tag,objective_relative_difference=gap,coefficient_relative_difference=relative(C,B),matched_objective=gap<=1e-6))
result=dict(records=len(records),expected_records=76,failed_certificates=sum(not r['passed'] for r in records),errors=errors,certificates=records,pairs=pairs)
(base/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('certificates','pairs')},indent=2))
assert len(records)==76 and result['failed_certificates']==0 and not errors
