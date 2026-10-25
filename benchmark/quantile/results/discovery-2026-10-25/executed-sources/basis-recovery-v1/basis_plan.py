import csv,json
from pathlib import Path
root=Path('.');existing=set()
for f in (root/'basis-results').glob('*-paths.csv'):
 for r in csv.DictReader(f.open()):existing.add((r['case'],r['seed'],r['rep'],r['method']))
plan=[]
for case in ['small','wide','ultrawide','tall100k']:
 seeds=[701,702,703] if case=='tall100k' else [1001,1002,1003]
 methods=['lp-dual-dual-warm','lp-dual-nullbasis-warm']
 if case=='ultrawide':methods+=['lp-split-primal-warm','lp-split-dual-warm']
 if case!='tall100k':methods+=['lp-split-nullbasis-primal','lp-split-nullbasis-dual']
 for seed in seeds:
  for rep in [1,2]:
   for method in methods:
    if (case,str(seed),str(rep),method) not in existing:plan.append((case,seed,rep,method))
with open('basis-recovery-plan.tsv','w') as f:
 for row in plan:f.write('\t'.join(map(str,row))+'\n')
with open('basis-recovery-tasks.tsv','w') as f:
 for case,seed in sorted({(r[0],r[1]) for r in plan}):f.write(f'{case}\t{seed}\n')
print(json.dumps({'planned_cells':len(plan),'existing_cells':len(existing),'reason':'Only missing observations after OOM, plus matched analytic starts for the split formulation; completed cap failures are retained rather than rerun.'},indent=2))
