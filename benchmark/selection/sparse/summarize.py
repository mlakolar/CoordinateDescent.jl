"""Summarize only retained Discovery records; never refit a model."""
import csv,json,os,statistics as st
from pathlib import Path
root=Path(os.environ['RUN']);out=root/'results'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(name,rows):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cases=read(root/'inputs/cases.csv');timings=[];accuracy=[];memory=[];memory_accuracy=[]
case_ids=[int(x) for x in os.environ.get('CASE_IDS','1,2,3,4,5,6,7,8,9,10,11').split(',')]
for rep in range(1,8):
 for engine in ('native','baseline','SLOPE.jl','R'):
  rows=read(out/f'timing-{engine}-{rep}.csv')
  engine_cases=[case for case in case_ids if engine!='baseline' or case!=11]
  expected=len(engine_cases)
  assert len(rows)==3*expected,(rep,engine,len(rows))
  for case in engine_cases:
   r=[x for x in rows if int(x['case'])==case];assert len(r)==3
   timings.append(dict(rep=rep,case=case,engine=engine,seconds=st.median(float(x['seconds']) for x in r),bytes=st.median(float(x['bytes']) for x in r) if engine!='R' else 'NA'))
 accuracy+=read(out/f'accuracy-{rep}.csv')
 memory_accuracy+=read(root/'memory'/f'accuracy-{rep}.csv')
 for case in (4,10,11):
  for engine in ('native','baseline','SLOPE.jl','R'):
   if case==11 and engine=='baseline':continue
   p=root/'memory'/f'rss-{engine}-{rep}-{case}.json';r=json.loads(p.read_text())
   assert r['exit_code']==0
   memory.append(dict(rep=rep,case=case,engine=engine,peak_rss_mib=r['peak_rss_kib']/1024))
assert len(accuracy)==7*sum(3 if case==11 else 4 for case in case_ids)
assert len(memory_accuracy)==77
assert max(float(x['relative_residual']) for x in memory_accuracy)<=1e-7
assert max(float(x['relative_objective_difference']) for x in memory_accuracy)<=1e-7
assert max(float(x['relative_residual']) for x in accuracy)<=1e-7
assert max(float(x['relative_objective_difference']) for x in accuracy)<=1e-7
write('memory-accuracy.csv',memory_accuracy);write('per-replication.csv',timings);write('accuracy-summary.csv',accuracy);write('memory-per-replication.csv',memory)
summary=[];ratios=[];memsummary=[]
for case in case_ids:
 for engine in ('native','baseline','SLOPE.jl','R'):
  vals=[r['seconds'] for r in timings if r['case']==case and r['engine']==engine]
  if not vals:continue
  summary.append(dict(case=case,name=cases[case-1]['name'],engine=engine,median_ms=1000*st.median(vals),min_ms=1000*min(vals),max_ms=1000*max(vals)))
  if engine!='native':
   pairs=[]
   for rep in range(1,8):
    a=next(r['seconds'] for r in timings if r['case']==case and r['engine']=='native' and r['rep']==rep)
    b=next(r['seconds'] for r in timings if r['case']==case and r['engine']==engine and r['rep']==rep)
    pairs.append(a/b)
   ratios.append(dict(case=case,comparison=f'native/{engine}',median=st.median(pairs),min=min(pairs),max=max(pairs)))
for case in (4,10,11):
 for engine in ('native','baseline','SLOPE.jl','R'):
  v=[r['peak_rss_mib'] for r in memory if r['case']==case and r['engine']==engine]
  if v:memsummary.append(dict(case=case,engine=engine,median_peak_rss_mib=st.median(v),min=min(v),max=max(v)))
write('timing-summary.csv',summary);write('paired-ratios.csv',ratios);write('memory-summary.csv',memsummary)
raw_native=read(out/'timing-native-1.csv')
validation={'timing_processes':28,'replications':7,'timing_rows':3*len(timings),'accuracy_rows':len(accuracy),'memory_accuracy_rows':len(memory_accuracy),'memory_max_relative_residual':max(float(x['relative_residual']) for x in memory_accuracy),'memory_processes':len(memory),'profiles':len(list(out.glob('profile-*.txt'))),'max_relative_residual':max(float(x['relative_residual']) for x in accuracy),'max_relative_objective_difference':max(float(x['relative_objective_difference']) for x in accuracy),'dense_controls_pass_5pct':all(next(r['median'] for r in ratios if r['case']==case and r['comparison']=='native/baseline')<=1.05 for case in (1,2,3)),'original_sparse_parity_with_SLOPE_jl':next(r['median'] for r in ratios if r['case']==4 and r['comparison']=='native/SLOPE.jl')<=1.0,'largest_design_bytes':next(int(float(r['design_bytes'])) for r in raw_native if int(r['case'])==11),'largest_workspace_bytes':next(int(float(r['workspace_bytes'])) for r in raw_native if int(r['case'])==11),'largest_dense_equivalent_bytes':10000*20000*8}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
lines=['# Discovery sparse SLOPE results','', 'Medians across seven process replications; each process contributes three timed batch medians.','', '| Case | Native ms | Baseline ms | SLOPE.jl ms | R ms |','|---|---:|---:|---:|---:|']
for case in case_ids:
 values=[]
 for engine in ('native','baseline','SLOPE.jl','R'):
  v=[r['median_ms'] for r in summary if r['case']==case and r['engine']==engine]
  values.append(f'{v[0]:.3f}' if v else 'omitted')
 lines.append('| '+cases[case-1]['name']+' | '+' | '.join(values)+' |')
lines+=['','Validation:','```json',json.dumps(validation,indent=2),'```','', 'Fresh-process peak RSS includes the interpreter, imports, JIT and fitting. It is not solely matrix storage. The 1.6 GB dense-equivalent case omits the old densifying backend by design.']
(out/'summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(validation));print('\n'.join(lines[:17]))
