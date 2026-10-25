"""Check saved fits under Slurm; shared sparse objective checks are untimed."""
import csv,hashlib,json,pathlib,statistics,sys,tomllib
import numpy as np
from scipy import sparse
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];base=root/'results'/phase
rows=[];errors=[]
for folder in sorted(base.glob('*-*')):
    if not folder.is_dir():continue
    m=tomllib.loads((root/'inputs'/folder.name/'meta.toml').read_text());n=m['n'];count=max(1,m['path'])
    input=root/'inputs'/folder.name
    y=np.fromfile(input/'y.bin',dtype='<f8')
    x=np.fromfile(input/'x.bin',dtype='<f8') if m['irregular'] else None
    assert hashlib.sha256((input/'y.bin').read_bytes()).hexdigest()==m['y_sha256']
    levels=np.array([m['lo']]) if count==1 else np.exp(np.linspace(np.log(m['hi']),np.log(m['lo']),count))
    D=sparse.eye(n,format='csc')
    for k in range(1,m['order']+2):
        if k>1 and x is not None:D=sparse.diags((k-1)/(x[k-1:]-x[:n-k+1]))@D
        A=sparse.diags([-np.ones(n-k),np.ones(n-k)],[0,1],shape=(n-k,n-k+1),format='csc')
        D=A@D
    variants={};timings={}
    for variant in ('baseline','candidate'):
        dest=folder/variant
        try:
            ts=list(csv.DictReader((dest/'timings.csv').open()));timings[variant]=statistics.median(float(r['seconds']) for r in ts)
            B=np.fromfile(dest/'coefficients.bin',dtype='<f8').reshape((n,count),order='F');variants[variant]=B
            actual=tomllib.loads((dest/'fit.toml').read_text());delta=D@B
            objective=(np.sum((B-y[:,None])**2,axis=0)/2+levels*np.sum(abs(delta),axis=0))/n
            err=float(np.max(abs(objective-actual['objectives'])/np.maximum(abs(objective),np.finfo(float).eps)))
            expected_df=np.minimum(n,m['order']+1+np.sum(abs(delta)>1e-4,axis=0))
            good=len(ts)==3 and all(r['converged']=='true' for r in ts) and np.all(np.isfinite(B)) and err<=1e-6 and np.array_equal(expected_df,actual['df'])
            if not good:errors.append(dict(input=folder.name,variant=variant,error='fit/objective/DF/convergence check failed'))
        except Exception as e:errors.append(dict(input=folder.name,variant=variant,error=repr(e)))
    if len(variants)==2:
        B,C=variants['baseline'],variants['candidate']
        ce=float(np.max(np.linalg.norm(B-C,axis=0)/np.maximum(np.linalg.norm(B,axis=0),np.finfo(float).eps)))
        bo=(np.sum((B-y[:,None])**2,axis=0)/2+levels*np.sum(abs(D@B),axis=0))/n
        co=(np.sum((C-y[:,None])**2,axis=0)/2+levels*np.sum(abs(D@C),axis=0))/n
        oe=float(np.max(abs(bo-co)/np.maximum(abs(bo),np.finfo(float).eps)))
        if ce>1e-4 or oe>1e-6:errors.append(dict(input=folder.name,error='before/after accuracy failed'))
        rows.append(dict(input=folder.name,baseline_seconds=timings['baseline'],candidate_seconds=timings['candidate'],
            ratio=timings['candidate']/timings['baseline'],coefficient_error=ce,objective_error=oe,bitwise_equal=bool(np.array_equal(B,C))))
result=dict(phase=phase,cases=len(rows),errors=errors,results=rows)
(base/'inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert not errors
