from pathlib import Path
import json,csv
import numpy as np
cases=['gaussian-moderate','gaussian-correlated','binomial-moderate','binomial-correlated']
checks=[]
for case in cases:
 for rep in [1,2,3]:
  old=Path('study')/f'holdout-{case}-{rep}.csv';new=Path('study-head')/f'holdout-{case}-{rep}.csv'
  a=[r for r in csv.DictReader(old.open()) if r['method']=='automatic' and int(r['rep'])==rep]
  b=[r for r in csv.DictReader(new.open()) if r['method']=='automatic' and int(r['rep'])==rep]
  assert len(a)==len(b)==1,(case,rep,len(a),len(b))
  def coefficients(f):
   return np.array([[float(r[4]),float(r[5])] for r in csv.reader(open(str(f)+'.coefficients')) if r[1]=='automatic' and int(r[0])==rep])
  aa,bb=coefficients(old),coefficients(new)
  checks.append(dict(case=case,rep=rep,lambda_old=float(a[0]['lambda']),lambda_head=float(b[0]['lambda']),node_old=float(a[0]['node_lambda']),node_head=float(b[0]['node_lambda']),beta_max_abs=float(np.max(abs(aa[:,0]-bb[:,0]))),se_max_abs=float(np.max(abs(aa[:,1]-bb[:,1]))),converged=b[0]['converged']))
Path('bridge-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
