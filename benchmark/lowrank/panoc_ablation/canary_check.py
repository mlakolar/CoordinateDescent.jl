import pathlib,json,tomllib,numpy as np
from scipy import linalg as la
root=pathlib.Path('results/prototype-canary-v1');s=tomllib.loads((root/'fits.toml').read_text());checks=[]
for r in s['records']:
 folder=pathlib.Path('inputs/pilot-v1')/r['case'];m=json.loads((folder/'meta.json').read_text());n,p,q=(m[k] for k in ['n','p','q'])
 X=np.fromfile(folder/'X.bin',dtype='<f8').reshape((n,p),order='F');Y=np.fromfile(folder/'Y.bin',dtype='<f8').reshape((n,q),order='F');X-=X.mean(0);Y-=Y.mean(0)
 B=np.fromfile(r['file'],dtype='<f8').reshape((p,q),order='F');R=Y-X@B;G=-X.T@R/n;L=m['L'];lam=r['lambda']
 U,s,Vt=la.svd(B-G/L,full_matrices=False);prox=(U*np.maximum(s-lam/L,0))@Vt
 mapping=L*la.norm(B-prox)/max(1.,la.norm(X.T@Y/n));obj=np.sum(R*R)/(2*n)+lam*la.svdvals(B).sum();dualR=R/n*min(1.,lam/max(la.svdvals(G)[0],np.finfo(float).tiny));dual=np.sum(Y*dualR)-n*np.sum(dualR*dualR)/2;gap=max(0.,obj-dual)/max(1.,abs(obj),abs(dual))
 rank=int(np.sum(la.svdvals(B)>min(B.shape)*np.finfo(float).eps*la.svdvals(B)[0]));passed=gap<=1e-6 and mapping<=1e-6 and abs(obj-r['objective'])<=1e-10*max(1.,abs(obj)) and r['rank']==rank
 checks.append(dict(**r,relative_gap=gap,normalized_mapping=mapping,objective_error=abs(obj-r['objective']),passed=bool(passed)))
(root/'certificates.json').write_text(json.dumps(checks,indent=2));print(json.dumps(dict(certificates=len(checks),failures=[r for r in checks if not r['passed']]),indent=2));assert len(checks)==66 and all(r['passed'] for r in checks)
