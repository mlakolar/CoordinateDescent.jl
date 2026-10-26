import csv,json,os,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve,eigvalsh
root=Path(os.environ['HDM_REFERENCE']);base=root.parent;rows=[]
for case in range(1,17):
    out=root/f'case-{case}';S=np.loadtxt(base/'objective-suite-v1'/f'case-{case}'/'covariance.csv',delimiter=',');p=len(S)
    identity=np.eye(p);jz=np.loadtxt(out/'Julia-Z.csv',delimiter=',');jd=np.loadtxt(out/'Julia-diagonal.csv',delimiter=',')
    for engine in ('native','Knockoffs','R'):
        if engine=='R':
            Z=np.loadtxt(out/'R-Z.csv',delimiter=',');K=np.loadtxt(out/'R-K0.csv',delimiter=',');K1=np.loadtxt(out/'R-K1.csv',delimiter=',')
            A=identity-(K1-K);D=S@A;d=np.diag(D);diagonal_error=np.linalg.norm(D-np.diag(d),np.inf)/max(1,np.linalg.norm(D,np.inf))
        else:
            Z=jz;K=np.loadtxt(out/f'Julia-{engine}.csv',delimiter=',');d=jd;A=solve(S,np.diag(d),assume_a='pos');diagonal_error=0.
        target=2*np.diag(d)-np.diag(d)@solve(S,np.diag(d),assume_a='pos')
        C=solve(Z,K);root_error=float(np.linalg.norm(C.T@C-target,np.inf)/max(1,np.linalg.norm(target,np.inf)))
        scale=np.sqrt(np.diag(S));R=S/(scale[:,None]*scale[None,:]);a=d/np.diag(S)
        minimum=float(eigvalsh(2*R-np.diag(a))[0])
        rows.append(dict(case=case,p=p,engine=engine,diagonal_error=float(diagonal_error),root_relative_error=root_error,
                         normalized_joint_min=minimum,valid=bool(diagonal_error<=1e-9 and root_error<=1e-8 and minimum>=-1e-9),
                         equicorrelated_diagonal_difference=float(np.linalg.norm(d-jd,np.inf)/max(1,np.linalg.norm(jd,np.inf)))))
with (root/'sampler-certificates.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'groups':[],'R_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('R-*-source.txt')}}
for engine in ('native','Knockoffs','R'):
    g=[r for r in rows if r['engine']==engine]
    summary['groups'].append(dict(engine=engine,checks=len(g),valid=sum(r['valid'] for r in g),
        max_root_error=max(r['root_relative_error'] for r in g),max_diagonal_error=max(r['diagonal_error'] for r in g),
        max_default_diagonal_difference=max(r['equicorrelated_diagonal_difference'] for r in g)))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
assert all(r['valid'] for r in rows if r['engine'] in ('native','R')),'required native/R sampler identity failed'
# The pinned upstream full sampler is assessed, never repaired or silently
# included in valid FDR rankings. Its usable diagonal optimizer is separate.
