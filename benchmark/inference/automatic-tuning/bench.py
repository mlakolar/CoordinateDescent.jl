import sys,time,warnings,json
from pathlib import Path
import numpy as np
from econml.sklearn_extensions.linear_model import DebiasedLasso
engine,label,rep=sys.argv[1:];base=Path('bench-data')/label/rep;out=Path('bench')/f'{engine}-{label}-{rep}'
c=json.loads((base/'config.json').read_text());X=np.loadtxt(base/'X.csv',delimiter=',');y=np.loadtxt(base/'y.csv',delimiter=',')
def fit(A,b):return DebiasedLasso(alpha=c['lambda_main'],alpha_cov=c['lambda_node'],max_iter=100000,tol=1e-9,random_state=77,n_jobs=1).fit(A,b)
try:
 rg=np.random.default_rng(42);fit(rg.normal(size=(80,12)),rg.normal(size=80))
 with warnings.catch_warnings(record=True) as w:
  warnings.simplefilter('always');t=time.perf_counter();m=fit(X,y);elapsed=time.perf_counter()-t
 np.savetxt(str(out)+'-coef.csv',np.column_stack([m.coef_,m.coef_stderr_]),delimiter=',')
 (Path(str(out)+'.meta')).write_text(f'status=ok\nfit_seconds={elapsed}\ntargets={X.shape[1]}\nwarnings='+ ' | '.join(str(a.message) for a in w)+'\n')
except Exception as e:Path(str(out)+'.meta').write_text('status=error\nerror='+str(e)+'\n')
