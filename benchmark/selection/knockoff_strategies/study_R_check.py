"""Certify saved matched R fits; no refitting; Slurm only."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
prefix=os.environ.get('HDM_R_PREFIX','R')
root=Path(os.environ['HDM_KNOCKOFF_STUDY']);sc,shard=map(int,sys.argv[1:3]);out=root/f'scenario-{sc}-shard-{shard}'

def arr(f,n):
    v=np.fromfile(f,dtype='<f8',count=n);assert len(v)==n;return v

def selection(b):
    p=len(b)//2;W=np.abs(b[:p])-np.abs(b[p:])
    for t in sorted(set(abs(W[W!=0]))):
        if (1+np.count_nonzero(W<=-t))/max(1,np.count_nonzero(W>=t))<=.1:return set(np.flatnonzero(W>=t))
    return set()

def cert(A,y,b):
    if not np.isfinite(b).all():return float('inf'),float('nan')
    n=len(y);g=A.T@(A@b-y)/n;z=b-g
    prox=np.sign(z)*np.maximum(abs(z)-.04,0)
    return float(np.linalg.norm(b-prox,np.inf)/max(1,np.linalg.norm(A.T@y/n,np.inf))),float(np.linalg.norm(A@b-y)**2/(2*n)+.04*np.linalg.norm(b,1))

native={}
with (out/'fits.bin').open('rb') as f:
    while True:
        h=np.fromfile(f,dtype='<i8',count=3)
        if not len(h):break
        seed,p,nm=map(int,h);native[seed]=arr(f,2*p*nm).reshape((2*p,nm),order='F')
rows=[]
with (out/'R-inputs.bin').open('rb') as inputs,(out/f'{prefix}-fits.bin').open('rb') as fits:
    while True:
        h=np.fromfile(inputs,dtype='<i8',count=8)
        if not len(h):break
        n,p,k,scenario,rep,dataset,seed,j=map(int,h);A=arr(inputs,n*2*p).reshape((n,2*p),order='F');y=arr(inputs,n)
        fh=tuple(map(int,np.fromfile(fits,dtype='<i8',count=3)));assert fh==(seed,p,j)
        b=arr(fits,2*p);bn=native[seed][:,j-1];res,obj=cert(A,y,b);nr,no=cert(A,y,bn)
        rows.append(dict(seed=seed,scenario=sc,shard=shard,dataset=dataset,strategy=('equicorrelated','mvr','maxent')[j-1],
              valid=res<=1e-7,residual=res,native_residual=nr,objective=obj,native_objective=no,
              relative_objective_difference=abs(obj-no)/max(1,abs(no)),
              relative_coefficient_difference=float(np.linalg.norm(b-bn,np.inf)/max(1,np.linalg.norm(bn,np.inf))),
              selection_disagreement=len(selection(b)^selection(bn))))
    assert not fits.read(1)
with (out/f'{prefix}-certificates.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(certificates=len(rows),valid=sum(r['valid'] for r in rows),failures=[r for r in rows if not r['valid']],
             max_residual=max(r['residual'] for r in rows),max_relative_objective_difference=max(r['relative_objective_difference'] for r in rows),
             max_relative_coefficient_difference=max(r['relative_coefficient_difference'] for r in rows),
             selection_disagreements=sum(r['selection_disagreement']>0 for r in rows))
(out/f'{prefix}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert all(r['valid'] for r in rows),'R comparator accuracy failures retained'
print(json.dumps(summary))
