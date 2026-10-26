import time,numpy as np,pyproximal as px
from scipy import linalg as la
from pyproximal.optimization.primal import ProximalGradient,ADMM
class StopSolve(Exception):
 def __init__(self,B,k,reason):self.B,self.k,self.reason=B.copy(),k,reason
class RegressionLoss(px.ProxOperator):
 def __init__(self,X,Y,H,C):
  super().__init__(hasgrad=True);self.X,self.Y,self.gram,self.C=X,Y,H,C;self.matrix_shape=C.shape;self.factors=None
 def __call__(self,x):
  B=x.reshape(self.matrix_shape);return np.sum((self.X@B-self.Y)**2)/(2*len(self.X))
 def grad(self,x):
  B=x.reshape(self.matrix_shape);g=self.gram@B-self.C if self.gram is not None else self.X.T@(self.X@B)/len(self.X)-self.C;return g.ravel()
 def prox(self,x,tau):
  if self.factors is None:
   U,s,Vt=la.svd(self.X/np.sqrt(len(self.X)),full_matrices=False);self.factors=(s*s,Vt)
  s2,Vt=self.factors;rhs=x.reshape(self.matrix_shape)+tau*self.C;return (rhs-Vt.T@((tau*s2/(1+tau*s2))[:,None]*(Vt@rhs))).ravel()
def setup(X,Y):
 mx,my=X.mean(0),Y.mean(0);Xc,Yc=X-mx,Y-my;n,p=Xc.shape;H=Xc.T@Xc/n if p<=n else None;C=Xc.T@Yc/n;L=la.svdvals(Xc)[0]**2/n
 return dict(X=Xc,Y=Yc,H=H,C=C,L=L,mx=mx,my=my,loss=RegressionLoss(Xc,Yc,H,C))
def svt(B,t):
 U,s,Vt=la.svd(B,full_matrices=False);return (U*np.maximum(s-t,0))@Vt
def solve(method,cache,lam,start):
 C,L,loss=cache['C'],cache['L'],cache['loss'];k=0;began=time.perf_counter();penalty=px.Nuclear(C.shape,sigma=lam)
 def check(x):
  nonlocal k
  k+=1;B=x.reshape(C.shape)
  if k%10==0:
   G=loss.grad(x).reshape(C.shape);mapping=L*la.norm(B-svt(B-G/L,lam/L))/max(1.,la.norm(C))
   if mapping<=1e-8:raise StopSolve(B,k,'converged')
  if time.perf_counter()-began>=60:raise StopSolve(B,k,'timeout60s')
 try:
  if method=='py_fista':x=ProximalGradient(loss,penalty,start.ravel().copy(),tau=1/L,niter=20000,acceleration='fista',tol=None,callback=check,show=False)
  else:
   _,z=ADMM(loss,penalty,start.ravel().copy(),tau=1/L,niter=20000,tol=None,callback=lambda x,z:check(z),callbackz=True,show=False);x=z
  B=x.reshape(C.shape);ok=False;reason='iteration_limit'
 except StopSolve as stop:B,k,reason=stop.B,stop.k,stop.reason;ok=reason=='converged'
 # Complete-result wrapper includes the same rank/support/intercept/DF/residual/objective fields.
 R=cache['Y']-cache['X']@B;s=la.svdvals(B);rk=int(np.sum(s>min(B.shape)*np.finfo(float).eps*s[0]));rows=np.flatnonzero(np.any(B!=0,axis=1));sx=la.svdvals(cache['X'][:,rows]) if len(rows) else np.array([]);xr=int(np.sum(sx>min(cache['X'].shape[0],len(rows))*np.finfo(float).eps*sx[0])) if len(sx) else 0
 return dict(B=B,intercept=cache['my']-B.T@cache['mx'],residual=R,objective=float(np.sum(R*R)/(2*len(R))+lam*s.sum()),rank=rk,rows=(rows+1).tolist(),df=rk*(xr+B.shape[1]-rk)+B.shape[1],iterations=k,converged=ok,reason=reason)
