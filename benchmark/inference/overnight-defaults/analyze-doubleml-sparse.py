import csv,json,math,runpy,sys
from pathlib import Path
stage,maxrep=sys.argv[1],int(sys.argv[2]);casefile='../doubleml-v2/doubleml-cases.csv';cases=list(csv.DictReader(open(casefile)))
rows=[];diagnostics=[];errors=[];notices=[];missing=[]
for c in cases:
 for rep in range(1,maxrep+1):
  for score in ['nuisance_space','instrument']:
   stem=Path('outputs')/stage/c['case']/(str(rep)+'.doubleml_sparse_'+score)
   if not Path(str(stem)+'.complete').exists():missing.append([c['case'],rep,score])
   for suffix,dest in [('csv',rows),('diagnostics.csv',diagnostics)]:
    p=Path(str(stem)+'.'+suffix)
    if p.exists():
     for r in csv.DictReader(p.open()):r.update(case=c['case'],rep=rep,family='binomial');dest.append(r)
   for suffix,dest in [('errors.json',errors),('warnings.json',notices)]:
    p=Path(str(stem)+'.'+suffix)
    if p.exists():dest.extend(dict(case=c['case'],rep=rep,score=score,**r) for r in json.loads(p.read_text()))
   p=Path(str(stem)+'.dataset.error')
   if p.exists():errors.append(dict(case=c['case'],rep=rep,score=score,error=p.read_text()))
dest=Path('reports')/(stage+'-doubleml-sparse');dest.mkdir(parents=True,exist_ok=True)
for name,data in [('records',rows),('diagnostics',diagnostics)]:
 if data:
  with (dest/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
keys=[(r['case'],r['rep'],r['method'],r['target']) for r in rows]
checks=dict(planned_datasets=len(cases)*maxrep,planned_rows=len(cases)*maxrep*4,rows=len(rows),unique_rows=len(set(keys)),errors=len(errors),warnings=len(notices),missing_markers=missing,max_se_parity=max((float(r['se_parity']) for r in diagnostics),default=None))
for name,data in [('checks',checks),('errors',errors),('warnings',notices)]:(dest/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(checks))
if missing or not rows or len(keys)!=len(set(keys)):raise SystemExit(1)
sys.argv=['summarize-slots.py',stage,'doubleml-sparse',str(maxrep),casefile];runpy.run_path('../doubleml-v2/summarize-slots.py',run_name='__main__')
