"""Independent original-design lasso/root/threshold check; Slurm only."""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve

root=Path(os.environ['HDM_KNOCKOFF_STUDY'])
sc,shard,count=map(int,sys.argv[1:4]);out=root/f'scenario-{sc}-shard-{shard}'
S=np.loadtxt(out/'supplied-covariance.csv',delimiter=',')
D=np.loadtxt(out/'diagonals.csv',delimiter=',');p=len(S);maps=[]
for j in range(3):
    d=D[:,j]
    if not np.isfinite(d).all():
        maps.append(None);continue
    A=solve(S,np.diag(d),assume_a='pos');C=2*np.diag(d)-np.diag(d)@A
    ev,U=np.linalg.eigh(C)
    assert ev[0]>=-1e-11*max(1,np.linalg.norm(C,np.inf)), 'infeasible conditional covariance'
    # Match the documented native upper Cholesky (or boundary eigen root).
    try:
        F=np.linalg.cholesky(C).T
    except np.linalg.LinAlgError:
        F=np.diag(np.sqrt(np.maximum(ev,0)))@U.T
    np.testing.assert_allclose(F.T@F,C,rtol=1e-10,atol=1e-12)
    maps.append((A,F))

def arr(f,count):
    v=np.fromfile(f,dtype='<f8',count=count)
    assert len(v)==count,'truncated record'
    return v

def threshold(W):
    for t in sorted(set(abs(W[W!=0]))):
        if (1+np.count_nonzero(W<=-t))/max(1,np.count_nonzero(W>=t))<=.1:
            return t
    return float('inf')

rows=[]
with (out/'inputs.bin').open('rb') as fi,(out/'fits.bin').open('rb') as ff:
    for dataset in range(1,count+1):
        n,p,k,scenario,rep,index,seed=map(int,np.fromfile(fi,dtype='<i8',count=7))
        assert (scenario,rep,index)==(sc,shard,dataset)
        X=arr(fi,n*p).reshape((n,p),order='F');y=arr(fi,n);Z=arr(fi,n*p).reshape((n,p),order='F');beta=arr(fi,p)
        sf,pf,nm=map(int,np.fromfile(ff,dtype='<i8',count=3));assert (sf,pf,nm)==(seed,p,3)
        B=arr(ff,2*p*3).reshape((2*p,3),order='F')
        for j,strategy in enumerate(('equicorrelated','mvr','maxent')):
            b=B[:,j];valid=bool(maps[j] is not None and np.isfinite(b).all())
            row=dict(seed=seed,scenario=sc,shard=shard,dataset=dataset,strategy=strategy,valid=False,
                     residual=float('inf'),objective=float('nan'),selected=0,true=0,false=0,
                     fdp=float('nan'),power=float('nan'),threshold=float('nan'),selected_ids='')
            if valid:
                A,F=maps[j];K=X-X@A+Z@F;design=np.column_stack((X,K))
                gradient=design.T@(design@b-y)/n
                prox=np.sign(b-gradient)*np.maximum(np.abs(b-gradient)-.04,0)
                residual=float(np.linalg.norm(b-prox,np.inf)/max(1,np.linalg.norm(design.T@y/n,np.inf)))
                W=np.abs(b[:p])-np.abs(b[p:]);t=threshold(W);selected=np.flatnonzero(W>=t) if np.isfinite(t) else np.array([],int)
                V=int(np.count_nonzero(selected>=k));R=len(selected)
                row.update(valid=bool(residual<=1e-7),residual=residual,
                           objective=float(np.linalg.norm(design@b-y)**2/(2*n)+.04*np.linalg.norm(b,1)),
                           selected=R,true=R-V,false=V,fdp=V/max(1,R),power=(R-V)/k if k else float('nan'),
                           threshold=t,selected_ids=';'.join(map(str,selected+1)))
            rows.append(row)
    assert not fi.read(1) and not ff.read(1),'unread extra records'
with (out/'certificates.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'scenario':sc,'shard':shard,'inputs':count,'certificates':len(rows),
         'valid':sum(r['valid'] for r in rows),'failures':[r for r in rows if not r['valid']],
         'max_residual':max(r['residual'] for r in rows)}
(out/'check-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert all(r['valid'] for r in rows),'independent certificate failures retained'
print(json.dumps(summary))
