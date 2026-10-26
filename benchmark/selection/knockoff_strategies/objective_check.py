"""Independent full-joint construction checks and convex oracles; Slurm only."""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.linalg import eigvalsh, solve
import cvxpy as cp

root=Path(os.environ['HDM_CONSTRUCTION_SUITE'])
allrows=[]
for case in range(1,17):
    out=root/f'case-{case}'
    S=np.loadtxt(out/'covariance.csv',delimiter=',')
    p=len(S);variance=np.diag(S);R=S/np.sqrt(variance[:,None]*variance[None,:])
    with (out/'diagonals.csv').open() as f:
        entries=list(csv.DictReader(f))
    for row in entries:
        d=np.fromstring(row['s'],sep=';');a=d/variance
        result=dict(case=case,p=p,source=row['source'],strategy=row['strategy'],setting=row['setting'],
                    returned=row['success']=='true',error=row['error'])
        result.update(valid=False,objective=np.nan,stationarity=np.nan,joint_min=np.nan,root_error=np.nan)
        if np.isfinite(a).all():
            D=np.diag(a);G=np.block([[R,R-D],[R-D,R]])
            eig=eigvalsh(G);result['joint_min']=float(eig[0])
            C=2*D-D@solve(R,D,assume_a='pos')
            ev,U=np.linalg.eigh(C);rootmap=np.diag(np.sqrt(np.maximum(ev,0)))@U.T
            result['root_error']=float(np.linalg.norm(rootmap.T@rootmap-C,np.inf)/max(1,np.linalg.norm(C,np.inf)))
            result['valid']=bool(np.min(a)>=-1e-12 and eig[0]>=-1e-11 and ev[0]>=-1e-11)
            if np.min(a)>0 and eig[0]>0:
                Q=solve(G,np.eye(2*p),assume_a='pos')
                if row['strategy']=='mvr':
                    result['objective']=float(np.trace(Q))
                    # Differentiate full G with respect to diagonal a_j.
                    Q2=Q@Q;gradient=2*np.diag(Q2[:p,p:])
                    denom=np.diag(Q2[:p,:p])+np.diag(Q2[p:,p:])
                elif row['strategy']=='maxent':
                    result['objective']=float(-np.linalg.slogdet(G)[1])
                    gradient=2*np.diag(Q[:p,p:]);denom=np.diag(Q[:p,:p])+np.diag(Q[p:,p:])
                else:
                    gradient=np.zeros(p);denom=np.ones(p)
                result['stationarity']=float(np.max(np.abs(gradient)/np.maximum(1,denom)))
        allrows.append(result)
    # Independent convex comparator only for the fixed small cases.
    if p<=30:
        for strategy in ('mvr','maxent'):
            a=cp.Variable(p);M=2*R-cp.diag(a)
            if strategy=='mvr':
                V=cp.Variable((p,p),symmetric=True)
                problem=cp.Problem(cp.Minimize(cp.trace(V)+cp.sum(cp.inv_pos(a))),
                    [a>=1e-10,cp.bmat([[M,np.eye(p)],[np.eye(p),V]])>>0])
            else:
                problem=cp.Problem(cp.Minimize(-cp.log_det(M)-cp.sum(cp.log(a))),[a>=1e-10])
            record=dict(case=case,p=p,source='CVXPY-CLARABEL',strategy=strategy,setting='oracle',returned=False,
                        valid=False,objective=np.nan,stationarity=np.nan,joint_min=np.nan,root_error=np.nan,error='')
            try:
                problem.solve(solver='CLARABEL',max_iter=500,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
                record['error']=str(problem.status)
                record['returned']=a.value is not None
                if a.value is not None:
                    av=a.value;D=np.diag(av);G=np.block([[R,R-D],[R-D,R]])
                    record['joint_min']=float(eigvalsh(G)[0]);record['valid']=bool(record['joint_min']>0 and np.min(av)>0)
                    if record['valid']:
                        Q=solve(G,np.eye(2*p),assume_a='pos')
                        record['objective']=float(np.trace(Q) if strategy=='mvr' else -np.linalg.slogdet(G)[1])
                    np.savetxt(out/f'oracle-{strategy}.csv',av,delimiter=',')
            except Exception as error:
                record['error']=str(error).replace('\n',' ')
            allrows.append(record)
with (root/'independent-objectives.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
summary={'records':len(allrows),'adapter_failures':[r for r in allrows if r['source']=='adapter' and not r['returned']],
         'infeasible_returns':[r for r in allrows if r['returned'] and not r['valid']],
         'numpy':np.__version__,'scipy':scipy.__version__,'cvxpy':cp.__version__,'solvers':cp.installed_solvers()}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Keep package-default failures as evidence, but required provisional/robust
# adapters must meet feasibility; objective gap acceptance is assessed explicitly.
required=[r for r in allrows if r['source']=='adapter' and r['setting'] in ('provisional','robust')]
assert len(required)==64
assert all(r['returned'] and r['valid'] for r in required), 'required adapter construction failed; retain all rows'
print(json.dumps(summary,indent=2))
