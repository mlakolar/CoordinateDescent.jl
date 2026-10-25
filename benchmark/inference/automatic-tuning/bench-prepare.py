"""Discovery only. New speed datasets, independent of development/holdout."""
import json,csv
from pathlib import Path
import numpy as np
root=Path.cwd(); prior=Path('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot')
for label,n,rho,s in [('moderate',200,.3,5),('correlated',100,.9,15)]:
    dev=[]
    for f in Path('study').glob('development-gaussian-'+label+'-*.csv'):
        dev.extend(csv.DictReader(f.open()))
    dev=[r for r in dev if r['method']=='automatic' and r['converged']=='true']
    assert len(dev)==100,(label,len(dev))
    lam=float(np.median([float(r['lambda']) for r in dev]));node=float(np.median([float(r['node_lambda']) for r in dev]))
    for rep in range(1,11):
        rg=np.random.default_rng(91400000+(1000 if label=='correlated' else 0)+rep)
        X=rg.normal(size=(n,500))
        for j in range(1,500):X[:,j]=rho*X[:,j-1]+np.sqrt(1-rho**2)*X[:,j]
        X-=X.mean(0);X/=np.sqrt(np.mean(X**2,0))
        beta=np.zeros(500);beta[:s]=.5*(-1.)**np.arange(s)
        y=.2+X@beta+rg.normal(size=n)
        d=Path('bench-data')/label/str(rep);d.mkdir(parents=True,exist_ok=True)
        np.savetxt(d/'X.csv',X,delimiter=',');np.savetxt(d/'y.csv',y,delimiter=',')
        (d/'config.json').write_text(json.dumps(dict(n=n,p=500,lambda_main=lam,lambda_node=node,mu=float(np.sqrt(2*np.log(500)/n)),seed=91400000+(1000 if label=='correlated' else 0)+rep)))
        with (d/'config.csv').open('w') as f:f.write('lambda_main,lambda_node,mu\n'+f'{lam},{node},{np.sqrt(2*np.log(500)/n)}\n')
with open('bench-tasks.tsv','w') as f:
    for label in ['moderate','correlated']:
        for rep in range(1,11):
            for engine in ['ours','hdi','econml','oursqp','sslasso','SIHR','oursqp10']:
                f.write(f'{engine}\t{label}\t{rep}\n')
print('BENCH_PREPARED:20 independent datasets; fixed penalties from development only')
