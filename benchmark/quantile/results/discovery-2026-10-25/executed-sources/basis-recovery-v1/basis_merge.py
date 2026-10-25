import csv,json
from pathlib import Path
root=Path('.');dest=root/'basis-completed';dest.mkdir(exist_ok=True)
for suffix in ['paths','solutions']:
 groups={};keys=set()
 for folder,mem,attempt in [('basis-results',4,'13423596'),('basis-recovery-results',8,'recovery')]:
  for f in (root/folder).glob(f'*-{suffix}.csv'):
   for r in csv.DictReader(f.open()):
    k=tuple(r[x] for x in ['case','seed','rep','method'])+((r['k'],) if suffix=='solutions' else ())
    if k in keys:raise RuntimeError(f'Duplicate observed cell {k}')
    keys.add(k);r['worker_mem_gb']=mem;r['attempt']=attempt
    groups.setdefault((r['case'],r['seed']),[]).append(r)
 for (case,seed),rows in groups.items():
  with (dest/f'{case}-{seed}-{suffix}.csv').open('w') as out:
   w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'combined_path_files':len(list(dest.glob('*-paths.csv'))),'source_folders':['basis-results','basis-recovery-results'],'raw_outputs_preserved':True},indent=2))
