import pathlib,json,numpy as np
from scipy import linalg as la
CASES={'small':(300,50,10),'medium':(300,50,100),'responses':(300,50,1000),'wide':(200,400,100),'ill':(500,100,100),'deficient':(300,80,100)}
root=pathlib.Path('inputs/pilot-v1')
for ci,(case,(n,p,q)) in enumerate(CASES.items()):
 folder=root/case;folder.mkdir(parents=True,exist_ok=True)
 assert not (folder/'meta.json').exists(),'refuse input overwrite'
 rng=np.random.default_rng(103260000+1000*ci)
 X=rng.standard_normal((n,p))
 if case=='ill':
  U,_=la.qr(X-X.mean(0),mode='economic');V,_=la.qr(rng.standard_normal((p,p)),mode='economic');X=(U*(np.sqrt(n)*np.geomspace(1,1e-4,p)))@V.T
 if case=='deficient':X[:,40:]=X[:,:40]
 X+=rng.uniform(-.5,.5,p);A=np.zeros((p,3));A[:10]=rng.standard_normal((10,3))/np.sqrt(3)
 Y=X@(A@rng.standard_normal((3,q)))+rng.standard_normal((n,q))+rng.uniform(-1,1,q)
 Xc,Yc=X-X.mean(0),Y-Y.mean(0)
 lm=float(la.svdvals(Xc.T@Yc/n)[0]);L=float(la.svdvals(Xc)[0]**2/n)
 for key,z in [('X',X),('Y',Y)]:np.asarray(z,dtype='<f8').ravel(order='F').tofile(folder/(key+'.bin'))
 (folder/'meta.json').write_text(json.dumps(dict(case=case,n=n,p=p,q=q,rep=0,seed=103260000+1000*ci,lambda_max=lm,L=L)))
 (folder/'dims.txt').write_text(f'{n} {p} {q} {lm} {L}\n')
