"""Summarize standalone fit-process RSS after external certificates pass."""
import csv, json, os, statistics
from pathlib import Path
root=Path(os.environ['RUN']);rows=[];accuracy=[]
for rep in range(1,8):
 with (root/'memory'/f'accuracy-{rep}.csv').open() as f:
  accuracy.extend(csv.DictReader(f))
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if engine=='baseline' and case==11:continue
   receipt=json.loads((root/'memory'/f'rss-{engine}-{rep}-{case}.json').read_text())
   assert receipt['exit_code']==0
   rows.append(dict(rep=rep,case=case,engine=engine,peak_rss_mib=receipt['peak_rss_kib']/1024))
assert len(rows)==len(accuracy)==77
assert max(float(x['relative_residual']) for x in accuracy)<=1e-7
assert max(float(x['relative_objective_difference']) for x in accuracy)<=1e-7
def write(name, data):
 with (root/'results'/name).open('w') as f:
  out=csv.DictWriter(f,fieldnames=list(data[0]));out.writeheader();out.writerows(data)
write('per-replication.csv',rows);write('accuracy.csv',accuracy)
summary=[]
for case in (4,10,11):
 for engine in ('native','baseline','SLOPE.jl','R'):
  v=[r['peak_rss_mib'] for r in rows if r['case']==case and r['engine']==engine]
  if v:summary.append(dict(case=case,engine=engine,median_peak_rss_mib=statistics.median(v),min=min(v),max=max(v)))
write('summary.csv',summary)
(root/'results'/'validation.json').write_text(json.dumps(dict(processes=77,replications=7,
 max_relative_residual=max(float(x['relative_residual']) for x in accuracy),
 max_relative_objective_difference=max(float(x['relative_objective_difference']) for x in accuracy),
 certificate_scope='All backends externally checked after measured process exit; SLOPE.jl process does not import HDM/CD.'),indent=2)+'\n')
print('FIT_ONLY_MEMORY_SUMMARY_DONE',flush=True)
