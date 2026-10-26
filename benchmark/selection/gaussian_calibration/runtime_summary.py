"""Paired runtime/default report from saved records; run under Slurm."""
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(os.environ['HDM_STUDY'])

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    out=ROOT/'report';out.mkdir(exist_ok=True)
    receipts=[json.loads(p.read_text()) for p in sorted((ROOT/'checks').glob('*.json'))]
    certs=[];timings=[];by_process=defaultdict(list);coeffs={};api=[]
    for p in sorted((ROOT/'checks').glob('*.csv')):certs.extend(csv.DictReader(p.open()))
    invalid=defaultdict(int)
    for c in certs:
        key=int(c['scenario']),int(c['rep']),int(c['seed']),c['engine'],c['method'],c['payload'],c['setting'],c['round']
        invalid[key]+=c['valid']!='True'
    for p in sorted((ROOT/'fits').glob('*.timings.csv')):
        engine=p.name.split('-')[0];sc=int(p.name.split('-')[1][2:]);rep=int(p.name.split('-')[2].split('.')[0][5:])
        for r in csv.DictReader(p.open()):
            row=dict(case=sc,rep=rep,engine=engine,**r);timings.append(row)
            if r['payload']=='weights':continue
            key=sc,rep,int(r['seed']),engine,r['method'],r['payload'],r['setting'],r['round']
            if not invalid[key]:by_process[sc,rep,int(r['seed']),engine,r['method'],r['payload'],r['setting']].append(float(r['seconds']))
    medians={k:float(np.median(v)) for k,v in by_process.items() if len(v)==3 and min(v)>0}
    paired=defaultdict(list);per_input=[]
    for (sc,rep,seed,engine,method,payload,setting),value in medians.items():
        reference=medians.get((sc,rep,seed,'native',method,payload,'public'))
        if reference is None:continue
        ratio=value/reference
        per_input.append(dict(case=sc,rep=rep,seed=seed,engine=engine,method=method,payload=payload,setting=setting,
            seconds=value,native_public_seconds=reference,ratio_to_native_public=ratio))
        paired[sc,engine,method,payload,setting].append((rep,seed,ratio))
    summary=[]
    for (sc,engine,method,payload,setting),values in sorted(paired.items()):
        reps=defaultdict(list)
        for rep,seed,value in values:reps[rep].append(np.log(value))
        logs=np.array([np.mean(a) for a in reps.values()])
        margin=float(t.ppf(.975,len(logs)-1)*logs.std(ddof=1)/np.sqrt(len(logs))) if len(logs)>1 else float('nan')
        summary.append(dict(case=sc,engine=engine,method=method,payload=payload,setting=setting,
          independent_processes=len(logs),inputs=len(values),geometric_ratio=float(np.exp(logs.mean())),
          lo=float(np.exp(logs.mean()-margin)),hi=float(np.exp(logs.mean()+margin)),
          min_input_ratio=min(x[2] for x in values),max_input_ratio=max(x[2] for x in values)))
    write_csv(out/'timings.csv',timings);write_csv(out/'paired-input-ratios.csv',per_input);write_csv(out/'paired-runtime.csv',summary)
    primary_failed=sum(c['valid']!='True' and c['setting']!='default' for c in certs)
    default_failed=sum(c['valid']!='True' and c['setting']=='default' for c in certs)
    cpu=list((ROOT/'fits').glob('*-profiles/*.cpu.txt'));alloc=list((ROOT/'fits').glob('*-profiles/*.alloc.txt'))
    empty_cpu=[str(p) for p in cpu if not p.stat().st_size or 'no samples' in p.read_text().lower()]
    resources=[json.loads(p.read_text()) for p in (ROOT/'logs').glob('*.resource.json')]
    result=dict(receipts=len(receipts),inputs=sum(r['inputs'] for r in receipts),certificates=len(certs),
      primary_failed_certificates=primary_failed,default_failed_certificates=default_failed,
      max_residual=max(float(c['residual']) for c in certs),
      public_prepared_bitwise=all(r['public_prepared_bitwise'] for r in receipts),
      positive_allocation_witnesses=sum(r['positive_allocation_witnesses'] for r in receipts),
      cpu_profiles=len(cpu),empty_cpu_profiles=empty_cpu,allocation_profiles=len(alloc),
      worker_resources=resources,
      interval='paired log-ratios, average three independent input logs within each fresh process replication; pointwise Student-t across seven process replications',
      timers='complete returned fit/path before coefficient extraction; native prepared-data control caches preprocessing only and returns the same complete fields; public/default/unscreened costs separate; R high-resolution wall timer',
      validation='every timed returned point independently checked; inaccurate fits excluded by input/workload; failed/default counts retained; no coefficient-only comparisons')
    if os.environ.get('HDM_GATE')=='main':assert len(receipts)==28 and result['inputs']==84
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
