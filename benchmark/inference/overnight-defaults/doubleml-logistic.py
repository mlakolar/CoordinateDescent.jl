"""Official DoubleML LPLR reference, two prespecified target roles per dataset.

The native nested cross-fitting and score are unchanged. Learners are explicit
choices because DoubleML does not provide default nuisance learners.
"""
import csv,json,math,sys,time,warnings
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegressionCV,LassoCV
from sklearn.exceptions import ConvergenceWarning
import doubleml as dml

case,shard,maxrep,stage,score=sys.argv[1:];shard=int(shard);maxrep=int(maxrep)
cases=list(csv.DictReader(open('../doubleml-v1/calibration-cases.csv')))
ci=next(i for i,c in enumerate(cases) if c['case']==case);spec=cases[ci]
method='doubleml_'+score
for rep in range(shard,maxrep+1,8):
    stem=Path('outputs')/stage/case/str(rep)
    out=Path(str(stem)+'.'+method)
    if Path(str(out)+'.complete').exists():continue
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        liu=spec['design']=='liu'
        def load(name):return np.loadtxt(stem/(({'X':'cal-X','y':'cal-y','beta':'cal-truth','targets':'cal-ts'}[name] if liu else name)+'.csv'),delimiter=',')
        X=load('X');y=load('y');truth=load('beta');targets=load('targets').astype(int)
        targets=targets[[0,-1]] # First requested role and last requested null role.
        rows=[];diagnostics=[];failures=[];notice=[]
        for target in targets:
            started=time.perf_counter();j=int(target)-1
            try:
                np.random.seed(96000000+ci*10000+rep*10+int(target))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    warnings.filterwarnings('error',category=ConvergenceWarning)
                    M=LogisticRegressionCV(Cs=10,cv=5,penalty='l1',solver='liblinear',scoring='neg_log_loss',tol=1e-8,max_iter=10000,n_jobs=1,random_state=0)
                    t=LassoCV(alphas=60,cv=5,tol=1e-8,max_iter=100000,n_jobs=1)
                    m=LassoCV(alphas=60,cv=5,tol=1e-8,max_iter=100000,n_jobs=1)
                    data=dml.DoubleMLData.from_arrays(np.delete(X,j,axis=1),y,X[:,j])
                    fit=dml.DoubleMLLPLR(data,M,t,m,n_folds=5,n_folds_inner=5,n_rep=1,score=score,error_on_convergence_failure=True)
                    fit.fit(n_jobs_cv=1)
                    notice.extend(dict(target=int(target),category=w.category.__name__,message=str(w.message)) for w in caught)
                est=float(fit.coef[0]);se=float(fit.se[0]);pv=float(fit.pval[0])
                psi=np.asarray(fit.psi).reshape(len(y),-1)[:,0]
                deriv=np.asarray(fit.psi_deriv).reshape(len(y),-1)[:,0]
                direct_se=float(np.sqrt(np.mean(psi**2)/len(y))/abs(np.mean(deriv)))
                if not (all(math.isfinite(z) for z in [est,se,pv,direct_se]) and se>0 and 0<=pv<=1):raise ValueError('nonfinite/invalid native inference')
                if abs(se-direct_se)>1e-8*max(1,se):raise ValueError('native score sandwich parity failure')
                if abs(np.mean(psi))>1e-6:raise ValueError('native score not solved')
                if abs(pv-2*ndtr(-abs(est/se)))>1e-10:raise ValueError('normal pvalue parity failure')
                rows.append(dict(method=method,target=int(target),truth=truth[j],estimate=est,se=se,lower=est-1.959963984540054*se,upper=est+1.959963984540054*se,pvalue=pv,converged='true',seconds=time.perf_counter()-started,refit_status='native_nested_crossfit',initial_size='',projection_parameter='',**{'lambda':''}))
                diagnostics.append(dict(target=int(target),score_mean=float(np.mean(psi)),score_derivative=float(np.mean(deriv)),se_parity=abs(se-direct_se),n=len(y),p=X.shape[1]))
            except Exception as e:failures.append(dict(target=int(target),error=repr(e)))
        for suffix,data in [('csv',rows),('diagnostics.csv',diagnostics)]:
            if data:
                with Path(str(out)+'.'+suffix).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        for suffix,data in [('errors.json',failures),('warnings.json',notice)]:Path(str(out)+'.'+suffix).write_text(json.dumps(data,indent=2)+'\n')
    except Exception as e:Path(str(out)+'.dataset.error').write_text(repr(e)+'\n')
    Path(str(out)+'.complete').write_text('attempted native reference; inspect errors and complete target count\n')
