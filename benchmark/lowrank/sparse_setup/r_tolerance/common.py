"""Frozen shared inputs and independent closed-form references; run under Slurm."""
import hashlib, json, pathlib, sys
import numpy as np
from scipy import linalg as la

CASES = dict(small=(300,50,10), medium=(300,50,100), responses=(300,50,1000),
             wide=(200,400,100), ill=(500,100,100), deficient=(300,80,100))
EPS=np.finfo(float).eps
def save(path, x): np.asarray(x,dtype='<f8').ravel(order='F').tofile(path)
def read(path, shape): return np.fromfile(path,dtype='<f8').reshape(shape,order='F').copy(order='F')
def load(folder):
    folder=pathlib.Path(folder); m=json.loads((folder/'meta.json').read_text())
    return m,read(folder/'X.bin',(m['n'],m['p'])),read(folder/'Y.bin',(m['n'],m['q']))
def factors(X,Y):
    A=X-X.mean(0); Z=Y-Y.mean(0)
    U,s,Vt=la.svd(A,full_matrices=False)
    keep=s>max(A.shape)*EPS*s[0]
    M=U[:,keep].T@Z
    H,h,Wt=la.svd(M,full_matrices=False)
    return dict(A=A,Z=Z,mx=X.mean(0),my=Y.mean(0),U=U[:,keep],s=s[keep],V=Vt[keep].T,
                H=H,h=h,Wt=Wt,xrank=int(keep.sum()),diagnostic_xrank=int((s>min(A.shape)*EPS*s[0]).sum()))
def coefficient(f,r):
    if r==0:return np.zeros((f['A'].shape[1],f['Z'].shape[1]))
    return (f['V']@(f['H'][:,:r]/f['s'][:,None]))@(f['h'][:r,None]*f['Wt'][:r])
def exact_df(f,r):
    q=f['Z'].shape[1]; ans=r*(f['xrank']+q-r)+q
    for i in range(r):
        for j in range(r,len(f['h'])):
            ratio=f['h'][j]/f['h'][i];gap=1-ratio*ratio
            if gap<=EPS:return float('inf')
            ans+=2*ratio*ratio/gap
    return ans
def generate(root, reps):
    root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True)
    for ci,(case,(n,p,q)) in enumerate(CASES.items()):
        for rep in reps:
            folder=root/f'{case}-{rep}'
            if folder.exists():raise RuntimeError(f'refuse to overwrite {folder}')
            folder.mkdir()
            seed=10226000+1000*ci+rep
            rng=np.random.default_rng(seed)
            X=rng.standard_normal((n,p))
            if case=='ill':
                U,_=la.qr(X-X.mean(0),mode='economic')
                V,_=la.qr(rng.standard_normal((p,p)),mode='economic')
                X=(U*(np.sqrt(n)*np.geomspace(1,1e-4,p)))@V.T
            elif case=='deficient':X[:,40:]=X[:,:40]
            X+=rng.uniform(-.5,.5,p)
            L=np.zeros((p,3)); L[:10]=rng.standard_normal((10,3))/np.sqrt(3)
            Y=X@(L@rng.standard_normal((3,q)))+rng.standard_normal((n,q))+rng.uniform(-1,1,q)
            save(folder/'X.bin',X);save(folder/'Y.bin',Y)
            f=factors(X,Y); objectives=[];dfs=[]
            for r in range(4):
                B=coefficient(f,r);save(folder/f'reference-{r}.bin',B)
                obj=float(np.sum((f['Z']-f['A']@B)**2)/(2*n))
                closed=(np.sum(f['Z']**2)-np.sum(f['h'][:r]**2))/(2*n)
                assert abs(obj-closed)<=1e-10*max(1,abs(obj))
                objectives.append(obj);dfs.append(exact_df(f,r))
            meta=dict(case=case,rep=rep,n=n,p=p,q=q,rank=3,seed=seed,reference_objectives=objectives,
                      exact_df=dfs,xrank=f['xrank'],diagnostic_xrank=f['diagnostic_xrank'])
            for name in ('X','Y'):
                b=(folder/f'{name}.bin').read_bytes()
                meta[f'{name}_sha256']=hashlib.sha256(b).hexdigest()
                meta[f'{name}_md5']=hashlib.md5(b).hexdigest()
            (folder/'meta.json').write_text(json.dumps(meta,indent=2))
            (folder/'dims.txt').write_text(f'{n}\n{p}\n{q}\n')
    (root/'grid.json').write_text(json.dumps(CASES,indent=2))
if __name__=='__main__':generate(sys.argv[1],[0] if sys.argv[2]=='pilot' else range(1,21))
