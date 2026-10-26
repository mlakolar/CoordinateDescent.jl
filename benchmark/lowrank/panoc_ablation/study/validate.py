import pathlib,json,tomllib,sys,numpy as np
from scipy import linalg as la
folder,out=map(pathlib.Path,sys.argv[1:3]);m=json.loads((folder/'meta.json').read_text());n,p,q=(m[k] for k in ['n','p','q'])
X=np.fromfile(folder/'X.bin',dtype='<f8').reshape((n,p),order='F');Y=np.fromfile(folder/'Y.bin',dtype='<f8').reshape((n,q),order='F');mx,my=X.mean(0),Y.mean(0);X=X-mx;Y=Y-my;C=X.T@Y/n;L=m['L'];checks=[]
fits=(tomllib.loads((out/'fits.toml').read_text()) if (out/'fits.toml').exists() else json.loads((out/'fits.json').read_text()))['records'];data=np.memmap(out/'fits.bin',dtype='<f8',mode='r')
for r in fits:
 offset=r['offset_bytes']//8;B=np.array(data[offset:offset+p*q]).reshape((p,q),order='F');intercept=np.array(data[offset+p*q:offset+p*q+q]);lam=r['lambda'];R=Y-X@B;G=-X.T@R/n
 U,s,Vt=la.svd(B-G/L,full_matrices=False);prox=(U*np.maximum(s-lam/L,0))@Vt
 mapping=L*la.norm(B-prox)/max(1.,la.norm(C));sigma=la.svdvals(B);obj=float(np.sum(R*R)/(2*n)+lam*sigma.sum());theta=R/n*min(1.,lam/max(la.svdvals(G)[0],np.finfo(float).tiny));dual=float(np.sum(Y*theta)-n*np.sum(theta*theta)/2);gap=max(0.,obj-dual)/max(1.,abs(obj),abs(dual))
 rk=int(np.sum(sigma>min(p,q)*np.finfo(float).eps*sigma[0]));rows=np.flatnonzero(np.any(B!=0,axis=1));sx=la.svdvals(X[:,rows]) if len(rows) else np.array([]);xrank=int(np.sum(sx>min(n,len(rows))*np.finfo(float).eps*sx[0])) if len(sx) else 0;df=rk*(xrank+q-rk)+q
 intercept_error=float(la.norm(intercept-(my-B.T@mx))/max(1.,la.norm(my)))
 accuracy_pass=gap<=1e-6 and mapping<=1e-6 and r['converged'];fields_pass=(not r['has_full_result']) or (abs(obj-r['objective'])<=1e-10*max(1.,abs(obj)) and r['rank']==rk and r['df']==df and r['rows']==(rows+1).tolist() and r['native_residual_error']<=1e-12 and intercept_error<=1e-10)
 checks.append(dict(mode=r['mode'],kind=r['kind'],fraction=r['fraction'],objective=obj,relative_gap=float(gap),normalized_mapping=float(mapping),native_converged=r['converged'],accuracy_pass=bool(accuracy_pass),result_fields_pass=bool(fields_pass),rank_reference=rk,df_reference=df,intercept_error=intercept_error,passed=bool(accuracy_pass and fields_pass)))
(out/'certificates.json').write_text(json.dumps(checks,indent=2));print(json.dumps(dict(points=len(checks),failures=[r for r in checks if not r['passed']]),indent=2));assert checks and all(r['passed'] for r in checks)
