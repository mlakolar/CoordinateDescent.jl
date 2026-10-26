import csv,json,os,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
root=Path(os.environ['HDM_KNOCKOFF_RUNTIME']);primary=root.parent/'runtime-main-v1';points=defaultdict(list);resources=[];allchecks=[]
for rep in range(1,8):
    for case in range(1,6):
        for variant in ('baseline','candidate'):
            out=root/f'{variant}-case-{case}-rep-{rep}'
            for input in (1,2,3):
                assert (out/f'input-{input}.bin').read_bytes()==(primary/out.name/f'input-{input}.bin').read_bytes(), 'sensitivity must reuse identical primary inputs'
            rows=list(csv.DictReader((out/'timings.csv').open()));assert len(rows)==45
            for r in rows:points[(variant,case,r['payload'],rep,int(r['input']))].append(float(r['seconds']))
            checks=list(csv.DictReader((out/'checks.csv').open()));assert all(r['valid']=='true' for r in checks);allchecks+=checks
            resource=json.loads((root/f'resource-{variant}-{case}-{rep}.json').read_text());assert resource['returncode']==0
            resources.append(resource)
    assert (root/f'legacy-check-{rep}.done').exists()
assert all(len(v)==3 for v in points.values())
rng=np.random.default_rng(106999001);indices=rng.integers(0,7,size=(10000,7));rows=[];replicates=[]
for case in range(1,6):
    payloads=sorted({k[2] for k in points if k[:2]==('baseline',case)})
    for payload in payloads:
        ratios=[]
        for rep in range(1,8):
            a=[np.median(points[('candidate',case,payload,rep,i)])/np.median(points[('baseline',case,payload,rep,i)]) for i in (1,2,3)]
            value=float(np.median(a));ratios.append(value);replicates.append(dict(case=case,payload=payload,rep=rep,candidate_first=(case+rep)%2==0,ratio=value))
        v=np.asarray(ratios);lo,hi=np.quantile(np.median(v[indices],axis=1),[.025,.975])
        rows.append(dict(case=case,payload=payload,ratio=float(np.median(v)),lo95=float(lo),hi95=float(hi),min_ratio=float(min(v)),max_ratio=float(max(v))))
for name,data in [('paired-ratios.csv',rows),('replication-ratios.csv',replicates)]:
    with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary=dict(purpose='alternating-order sensitivity prompted by primary variant-phase confounding; primary results preserved',
 inputs_reused=105,extra_independent_datasets=0,fresh_processes=70,successful_resource_receipts=len(resources),legacy_assertions=315,
 certificates=len(allchecks),max_filter_residual=max(float(r['residual']) for r in allchecks if r['payload']=='filter' and r['round']!='0'),comparisons=rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Alternating-order sensitivity','',summary['purpose']+'. The same105 primary inputs are reused in70 fresh processes; this is not another independent data cohort. Baseline/candidate order alternates by(case+rep)parity, both run serially on the same allocated CPU per pair. Only equicorrelation is measured. Three rounds per input/three inputs per process, seven independent process pairs, bootstrap10000 seed106999001. Primary data and all primary timing rows are retained separately. No profiling is repeated.','',
 '| Case | Payload | Candidate / baseline | 95% process interval | Process range |','|---|---|---:|---|---|']
for r in rows:lines.append(f"| {r['case']} | {r['payload']} | {r['ratio']:.3f} | [{r['lo95']:.3f}, {r['hi95']:.3f}] | {r['min_ratio']:.3f}–{r['max_ratio']:.3f} |")
lines+=['','The diagonal row compares the new validated factory with the unchanged legacy vector selector; its additional work is not an old-selector regression. The other rows compare the same public interfaces. Small differences and adverse intervals remain visible. No pooling with primary timings or selection of favorable rounds.','']
(root/'TABLES.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'}))
