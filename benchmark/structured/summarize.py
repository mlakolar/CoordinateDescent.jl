# Run on Discovery inside a Slurm allocation, from a run root or recorded/.
import csv,json,statistics
from pathlib import Path
r=Path('.')
def rows(pattern):
 out=[]
 for p in sorted(r.glob(pattern)):
  with p.open() as f: out.extend(csv.DictReader(f,delimiter='\t'))
 return out
signal=rows('results/timings-*.tsv')
fused=[x for x in rows('results/regression-*.tsv') if x['family']=='fused']
admm=rows('results-fixed/regression-*.tsv')
assert len(signal)==108 and len(fused)==36 and len(admm)==72,(len(signal),len(fused),len(admm))
assert all(x['converged']=='true' for x in signal+fused+admm)
def summarize(data,keys):
 groups={}
 for x in data: groups.setdefault(tuple(x[k] for k in keys),[]).append(x)
 out=[]
 for key,v in groups.items():
  assert len(v)==9 and {x['seed'] for x in v}=={'8101','8102','8103'}
  assert all(sum(x['seed']==s for x in v)==3 for s in ('8101','8102','8103'))
  out.append(dict(zip(keys,key),measurements=len(v),median_seconds=statistics.median(float(x['seconds']) for x in v),min_seconds=min(float(x['seconds']) for x in v),max_seconds=max(float(x['seconds']) for x in v),median_allocated_bytes=statistics.median(int(x['bytes']) for x in v),min_iterations=min(int(x['iterations']) for x in v),max_iterations=max(int(x['iterations']) for x in v)))
 return out
out={'seeds':[8101,8102,8103],'rounds':3,'converged_signal_measurements':len(signal),'converged_regression_measurements':len(fused+admm),'signal':summarize(signal,['n','order']),'regression':summarize(fused+admm,['n','p','family','path']),'recovery':'Fused measurements reused from first regression attempt. Graph/trend measurements rerun after data-scaled ADMM default fix; failed attempt retained.'}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
