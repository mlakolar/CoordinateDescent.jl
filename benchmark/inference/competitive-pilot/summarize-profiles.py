import csv,json,statistics
from pathlib import Path
from collections import defaultdict
rows=[]; totals=defaultdict(list)
for p in sorted(Path('profiles').glob('*-total.csv')):
 rep=int(p.name.split('-')[-2])
 for r in csv.DictReader(open(p)):
  rows.append(dict(rep=rep,**r))
  totals[r['fun'].strip('"')].append(float(r['total.pct']))
with open('profile-totals.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={k:dict(replications=len(v),median_percent=statistics.median(v),min_percent=min(v),max_percent=max(v)) for k,v in totals.items()}
Path('profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ('calculate.Z','preprocess.group.testing','initial.estimator','cv.glmnet','glmnet','lasso.proj')},indent=2))
