import csv,json,statistics,math,collections
from pathlib import Path
N=Path('reports/gaussian-native-full1000');D=Path('reports/gaussian-core-comparison1000');fits=json.loads((N/'fits.json').read_text());rr=json.loads((N/'resource-records.json').read_text());groups=collections.defaultdict(list)
for filename,own in [(N/'records.csv',False),(Path('reports/seedfix-campaign-gaussian/records.csv'),True)]:
 for r in csv.DictReader(filename.open()):groups[(r['case'],'Julia_current' if own else r['engine'],int(r['rep']),'active' if float(r['truth'])!=0 else 'null')].append(r)
def stats(v):return dict(mean=statistics.mean(v),mcse=statistics.stdev(v)/math.sqrt(len(v)))
def metric(rs,key):
 if key=='coverage':return statistics.mean(float(r['lower'])<=float(r['truth'])<=float(r['upper']) for r in rs)
 if key=='rejection':return statistics.mean(float(r['pvalue'])<.05 for r in rs)
 return statistics.mean(float(r['upper'])-float(r['lower']) for r in rs)
pairs=[];timing=[]
for c in ['gaussian-moderate','gaussian-correlated','gaussian-benign']:
 for e in ['hdi','EconML','sslasso','SIHR','hdm']:
  ff=[r for r in fits if r['case']==c and r['engine']==e];resources=[r for r in rr if r['case']==c and r['engine']==e]
  assert len(ff)==len(resources)==1000
  timing.append(dict(case=c,engine=e,n=1000,targets=(100 if c=='gaussian-benign' else 500) if e in ['hdi','EconML','sslasso'] else 10,fit_seconds=stats([r['fit_seconds'] for r in ff]),process_seconds=stats([r['process_seconds'] for r in resources]),cpu_hours=sum(r['cpu_seconds'] for r in resources)/3600,max_worker_rss_gib=max(r['max_rss_kib'] for r in resources)/1024**2))
  for role in ['active','null']:
   s=dict(case=c,engine=e,role=role,paired_datasets=1000,scope='Julia minus finite native,includingwarned/uncertifiednative;strictavailabilityseparate')
   for k in ['coverage','rejection','length']:s[k]=stats([metric(groups[(c,'Julia_current',rep,role)],k)-metric(groups[(c,e,rep,role)],k) for rep in range(1,1001)])
   pairs.append(s)
(D/'paired-differences.json').write_text(json.dumps(pairs,indent=2)+'\n');(D/'native-timing-memory.json').write_text(json.dumps(timing,indent=2)+'\n')
lines=['\n## Native pipeline timing and memory\n','Native own tuning included. Julia all-slope20dataset timing is reported separately because its ten-target1000arm has a different target count. Per-process peakRSS,not totaljobmemory.\n','|Design|Native method|Inferred slopes|Mean fit ± datasetMCSE (s)|Peak workerRSS (GiB)|','|---|---|---:|---:|---:|']
for r in timing:lines.append(f"|{r['case']}|{r['engine']}|{r['targets']}|{r['fit_seconds']['mean']:.3f} ±{r['fit_seconds']['mcse']:.3f}|{r['max_worker_rss_gib']:.3f}|")
lines+=['','Paired dataset-level differences andMCSE for coverage,rejection and length are retained in paired-differences.json. No independent-target approximation is used. Native convergence/certificate caveats apply to these conditional differences.']
with (D/'RESULTS.md').open('a') as f:f.write('\n'.join(lines)+'\n')
