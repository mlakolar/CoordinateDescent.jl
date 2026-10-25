import csv,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
stage=sys.argv[1];maxrep=int(sys.argv[2]);cases=list(csv.DictReader(open('cases.csv')));groups=defaultdict(lambda:defaultdict(list));missing=[];errors=[];warnings=[];records=[]
for case in cases:
 c=case['case'];root=Path('outputs')/stage/c
 for rep in range(1,maxrep+1):
  stem=root/str(rep)
  for suffix in ['.jcomplete','.rcomplete']:
   if not stem.with_suffix(suffix).exists():missing.append([c,rep,suffix])
  for engine in ['julia','R','supp']:
   f=stem.with_suffix('.'+engine+'.csv')
   if not f.exists():continue
   for r in csv.DictReader(f.open()):
    r.update(case=c,rep=rep,family=case['family']);records.append(r)
    groups[(c,r['method'],'active' if float(r['truth']) else 'null')][rep].append(r)
 for f in root.glob('*.error'):errors.append({'case':c,'file':f.name,'error':f.read_text()})
 for f in root.glob('*.warnings'):warnings.append({'case':c,'file':f.name,'warnings':f.read_text()})
summary=[]
for key,reps in sorted(groups.items()):
 d=dict(zip(['case','method','target_class'],key));d['valid_datasets']=len(reps);d['planned_datasets']=maxrep
 metrics=defaultdict(list)
 for rep,rr in reps.items():
  metrics['coverage'].append(statistics.mean([float(float(r['lower'])<=float(r['truth'])<=float(r['upper'])) for r in rr]))
  metrics['rejection'].append(statistics.mean([float(float(r['pvalue'])<.05) for r in rr]))
  metrics['length'].append(statistics.mean([float(r['upper'])-float(r['lower']) for r in rr]))
  metrics['signed_z'].append(statistics.mean([(float(r['estimate'])-float(r['truth']))/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr]))
  metrics['seconds'].append(float(rr[0]['seconds']))
  metrics['converged'].append(float(all(r['converged'].lower()=='true' for r in rr)))
  metrics['refit_fallback'].append(float(any(r['refit_status'].startswith('fallback') for r in rr)))
 for name,v in metrics.items():d[name]=statistics.mean(v);d[name+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
 summary.append(d)
keys=[(r['case'],r['rep'],r['method'],r['target']) for r in records]
checks={'stage':stage,'cases':len(cases),'datasets':len(cases)*maxrep,'target_records':len(records),'unique_target_records':len(set(keys)),'missing_dataset_markers':missing,'method_error_count':len(errors),'warning_files':len(warnings),'finite_successful_outputs':all(all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) and float(r['se'])>0 for r in records)}
root=Path('reports')/(stage+'-supp');root.mkdir(parents=True,exist_ok=True)
for name,content in [('summary',summary),('checks',checks),('errors',errors),('warnings',warnings)]: (root/(name+'.json')).write_text(json.dumps(content,indent=2))
with (root/'records.csv').open('w') as f:
 if records:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
lines=['# Overnight development pilot','', 'EMPIRICAL. Success-conditioned method metrics must be read with valid/planned counts and errors. No failed methods are silently dropped from eligibility. Same data are reused across methods. Dataset-level MCSE, not independent-target MCSE.','', '|Case|Method|Targets|Valid/planned|Coverage|Rejection|Length|Seconds|','|---|---|---|---|---|---|---|---|']
for r in summary:lines.append(f"|{r['case']}|{r['method']}|{r['target_class']}|{r['valid_datasets']}/{r['planned_datasets']}|{r['coverage']:.4f}|{r['rejection']:.4f}|{r['length']:.4f}|{r['seconds']:.3f}|")
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(checks))
# Canary gates transport/data integrity, not the scientific success of every candidate.
if missing or len(keys)!=len(set(keys)) or not checks['finite_successful_outputs']:raise SystemExit(1)
for c in cases:
 if not any(k[0]==c['case'] and k[1]=='rldpe' for k in groups):raise SystemExit('no baseline for '+c['case'])
