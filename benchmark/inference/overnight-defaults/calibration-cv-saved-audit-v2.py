import csv,json,math,statistics
from pathlib import Path
rows=[];diffs=[]
for casepath in sorted(Path('outputs/calibration-cv-bounded').iterdir()):
 c=casepath.name
 for rep in range(1,6):
  p=Path('outputs/calibration-cv-bounded')/c/str(rep)
  if not p.exists():continue
  rows.extend(dict(r,case=c,rep=rep) for r in csv.DictReader((p/'julia-cv-checks.csv').open()))
  a=list(csv.reader((p/'julia-errors.csv').open()));b=list(csv.reader((p/'rcal-errors.csv').open()))
  for i,(ar,br) in enumerate(zip(a,b),1):
   for k,(av,bv) in enumerate(zip(ar,br),1):
    try:x,y=float(av),float(bv)
    except ValueError:continue
    if math.isfinite(x) and math.isfinite(y):diffs.append(dict(case=c,rep=rep,level=i,fold=k,julia=x,rcal=y,difference=x-y,relative=abs(x-y)/max(1,abs(x),abs(y))))
D=Path('reports/calibration-cv-saved-audit');D.mkdir(exist_ok=True,parents=True)
mismatch=[r for r in rows if int(r['index'])!=int(r['reference_index'])]
(D/'selection-disagreements.json').write_text(json.dumps(mismatch,indent=2)+'\n')
(D/'worst-fold-differences.json').write_text(json.dumps(sorted(diffs,key=lambda r:abs(r['difference']),reverse=True)[:30],indent=2)+'\n')
(D/'checks.json').write_text(json.dumps(dict(policy_rows=len(rows),selection_disagreements=len(mismatch),common_finite_fold_values=len(diffs),max_absolute_fold_difference=max(abs(r['difference']) for r in diffs),max_relative_fold_difference=max(r['relative'] for r in diffs)),indent=2)+'\n')
print('Policy rows',len(rows),'selection differences',len(mismatch))
