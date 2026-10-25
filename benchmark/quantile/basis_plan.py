import csv,json
from pathlib import Path
root=Path('.');existing={};counts={}
for f in (root/'basis-results').glob('*-paths.csv'):
 for r in csv.DictReader(f.open()):existing[(r['case'],r['seed'],r['rep'],r['method'])]=r
for f in (root/'basis-results').glob('*-solutions.csv'):
 for r in csv.DictReader(f.open()):
  k=(r['case'],r['seed'],r['rep'],r['method']);counts[k]=counts.get(k,0)+1
plan=[]
for case in ['small','wide','ultrawide','tall100k']:
 seeds=[701,702,703] if case=='tall100k' else [1001,1002,1003]
 methods=['lp-dual-dual-warm','lp-dual-nullbasis-warm']
 if case=='ultrawide':methods+=['lp-split-primal-warm','lp-split-dual-warm']
 if case!='tall100k':methods+=['lp-split-nullbasis-primal','lp-split-nullbasis-dual']
 for seed in seeds:
  for rep in [1,2]:
   for method in methods:
    k=(case,str(seed),str(rep),method)
    if k not in existing or counts.get(k,0)<int(existing[k]['levels']):plan.append((case,seed,rep,method))
with open('basis-recovery-plan.tsv','w') as f:
 for row in plan:f.write('\t'.join(map(str,row))+'\n')
with open('basis-recovery-tasks.tsv','w') as f:
 for case,seed in sorted({(r[0],r[1]) for r in plan}):f.write(f'{case}\t{seed}\n')
print(json.dumps({'planned_cells':len(plan),'existing_cells':len(existing),'reason':'Only missing observations after OOM, plus matched analytic starts for the split formulation; completed cap failures are retained rather than rerun.'},indent=2))
