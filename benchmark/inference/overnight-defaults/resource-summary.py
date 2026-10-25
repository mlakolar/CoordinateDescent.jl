"""Compute-node resource summary for completed stage prefixes; never refits."""
import csv,json,math,statistics,sys
from pathlib import Path
prefix,report=sys.argv[1:];dest=Path(report);dest.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(Path('resources').glob(prefix+'-*.txt')):
 if prefix=='gaussian-confirmation' and p.name.startswith('gaussian-confirmation-canary-'):continue
 d=dict(line.split('=',1) for line in p.read_text().splitlines() if '=' in line)
 if not all(k in d for k in ['process_seconds','max_rss_kib','cpu_seconds','process_exit']):raise ValueError('incomplete resource receipt '+str(p))
 rows.append(dict(file=str(p),**{k:float(v) for k,v in d.items()}))
if not rows:raise ValueError('no resource receipts for '+prefix)
summary=dict(prefix=prefix,workers=len(rows),nonzero_process_exits=sum(r['process_exit']!=0 for r in rows),cpu_seconds=sum(r['cpu_seconds'] for r in rows),worker_process_seconds_sum=sum(r['process_seconds'] for r in rows),max_worker_rss_gib=max(r['max_rss_kib'] for r in rows)/(1024**2),median_worker_rss_gib=statistics.median(r['max_rss_kib'] for r in rows)/(1024**2),note='Linux child-process peak RSS per worker, not simultaneous total job memory. Worker process seconds include startup/compilation and multiple datasets where sharded. CPU sum is consumed worker CPU, not allocated CPU walltime.')
records=dest/'records.csv'
if prefix=='gaussian-confirmation' and records.exists():
 groups={}
 for r in csv.DictReader(records.open()):groups.setdefault(r['case'],{})[int(r['rep'])]=float(r['seconds'])
 fits=[]
 for case,rr in sorted(groups.items()):
  for phase in ['first_per_worker','warm']:
   v=[s for rep,s in rr.items() if (rep<=8)==(phase=='first_per_worker')]
   if v:fits.append(dict(case=case,phase=phase,datasets=len(v),mean_seconds=statistics.mean(v),median_seconds=statistics.median(v),max_seconds=max(v),note='Inside-fit timer: initial CV+projection+HC3; first fit includes compilation after imports, not complete fresh-process startup.'))
 (dest/'fit-time-summary.json').write_text(json.dumps(fits,indent=2)+'\n')
(dest/'resource-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
