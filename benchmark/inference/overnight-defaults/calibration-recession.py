"""Numerical recession witnesses for convex calibrated logistic loss; compute nodes only."""
import csv,json,sys,math,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
case,rs=sys.argv[1:];root=Path('outputs/calibration-cv-engine')/case/rs;dest=Path('outputs/calibration-recession')/case/rs;dest.mkdir(parents=True,exist_ok=True)
base=Path('outputs/development')/case/rs;liu=case.startswith('liu-')
X=np.loadtxt(base/('cal-X.csv' if liu else 'X.csv'),delimiter=',');y=np.loadtxt(base/('cal-y.csv' if liu else 'y.csv'),delimiter=',')
rows=[];witness=[]
def analyze(N,yy,w,levels,scope,fold):
 W=w/w.mean();M=np.column_stack((np.ones(len(yy)),N));p=M.shape[1]
 Q=np.column_stack((M,-M));positive=yy==1
 A=np.vstack((-Q[positive],np.ones((1,2*p))));bound=np.r_[np.zeros(positive.sum()),1.]
 linear=(W*(1-yy))@M/len(yy)
 for a,lam in enumerate(levels,1):
  penalty=np.r_[0.,np.full(p-1,lam)];c=np.r_[linear+penalty,-linear+penalty]
  with warnings.catch_warnings(record=True) as notices:
   result=linprog(c,A_ub=A,b_ub=bound,bounds=(0,None),method='highs',options={'threads':1})
  r=dict(case=case,rep=int(rs),scope=scope,fold=fold,level=a,penalty=float(lam),lp_success=bool(result.success),lp_status=int(result.status),lp_message=result.message)
  if result.success:
   v=result.x[:p]-result.x[p:];u=M@v
   # Move the intercept very slightly upward to give a strictly positive
   # direction for every y=1 observation, then check in original coordinates.
   v[0]+=max(0.,1e-9-float(u[positive].min()));u=M@v
   slope=float(np.mean(W*(1-yy)*u)+lam*np.abs(v[1:]).sum());minimum=float(u[positive].min())
   found=minimum>=0 and slope < -1e-8
   r.update(recession_slope=slope,minimum_positive_response_direction=minimum,l1_direction_norm=float(np.abs(v).sum()),unbounded_witness=bool(found))
   if found and not any(z['scope']==scope and z['fold']==fold for z in witness):witness.append(dict(scope=scope,fold=fold,level=a,penalty=float(lam),recession_slope=slope,direction=v.tolist()))
  rows.append(r)
# Actual target-specific calibrated weights; offsets affect level values but
# do not affect recession directions. Analyze full sample and each training fold.
if (root/'j.csv').exists():
 j=int(np.loadtxt(root/'j.csv',delimiter=','))-1;N=np.delete(X,j,axis=1);w=np.loadtxt(root/'weight.csv',delimiter=',');folds=np.loadtxt(root/'foldid.csv',delimiter=',',dtype=int);levels=np.loadtxt(root/'levels.csv',delimiter=',')
 analyze(N,y,w,levels,'actual_cv',0)
 for k in range(1,6):
  tr=folds!=k;analyze(N[tr],y[tr],w[tr],levels,'actual_cv',k)
# Reconcile the three uncertified native fixed-penalty references too.
froot=Path('outputs/calibration-engine')/case/rs
if (froot/'weights.csv').exists():analyze(X[:,1:],y,np.loadtxt(froot/'weights.csv',delimiter=','),np.array([2.,.7])*np.sqrt(np.log(X.shape[1])/len(y)),'fixed_reference',0)
(dest/'checks.json').write_text(json.dumps(rows,indent=2)+'\n');(dest/'witnesses.json').write_text(json.dumps(witness,indent=2)+'\n')
print(json.dumps(dict(case=case,rep=int(rs),linear_programs=len(rows),witnesses=sum(r.get('unbounded_witness',False) for r in rows))))
