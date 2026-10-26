"""Native EconML defaults, two public solver levels unmodified; matched X/y."""
import csv,json,time,sys,warnings,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import norm
from econml.sklearn_extensions.linear_model import DebiasedLasso
_,case,rs=sys.argv[1:];rep=int(rs);base=Path('competitor-data')/case/rs
out=Path('outputs/gaussian-native-full')/case/f'{rep}.EconML';out.parent.mkdir(parents=True,exist_ok=True)
X=np.loadtxt(base/'X.csv',delimiter=',');y=np.loadtxt(base/'y.csv',delimiter=',');truth=np.loadtxt(base/'truth.csv',delimiter=',');ts=np.loadtxt(base/'targets.csv',dtype=int)-1
hashes=dict(l.split('=',1) for l in (base/'hashes.txt').read_text().splitlines())
assert hashlib.sha256(X.astype('<f8').tobytes(order='F')).hexdigest()==hashes['X']
assert hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()==hashes['y']
status={}
try:
 rng=np.random.default_rng(177);DebiasedLasso(random_state=77,n_jobs=1).fit(rng.normal(size=(80,12)),rng.normal(size=80))
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');t=time.perf_counter();f=DebiasedLasso(random_state=770000+rep,n_jobs=1).fit(X,y);elapsed=time.perf_counter()-t
 lo,hi=f.coef__interval(alpha=.05);se=f.coef_stderr_;bh=f.coef_
 with Path(str(out)+'.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=['method','target','truth','estimate','se','lower','upper','pvalue','converged','seconds','refit_status','initial_size','lambda','projection_parameter']);writer.writeheader()
  for j in ts:writer.writerow(dict(method='EconML',target=j+1,truth=truth[j],estimate=bh[j],se=se[j],lower=lo[j],upper=hi[j],pvalue=2*norm.sf(abs(bh[j]/se[j])),converged=True,seconds=elapsed,refit_status='native_default',initial_size='',**{'lambda':'','projection_parameter':''}))
 status=dict(status='ok',fit_seconds=elapsed,actual_inferred_targets=X.shape[1],warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in ws])
except Exception as e:status=dict(status='error',error=repr(e))
Path(str(out)+'.json').write_text(json.dumps(status,indent=2)+'\n');Path(str(out)+'.complete').write_text('attempted; inspect warnings/status\n')
