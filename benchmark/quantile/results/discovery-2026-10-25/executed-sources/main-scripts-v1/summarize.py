"""Run on Discovery; aggregation never reruns a fit."""
import csv, json, math, statistics as st, sys
from pathlib import Path
from collections import defaultdict
root=Path(sys.argv[1]); dest=Path(sys.argv[2]); dest.mkdir(parents=True,exist_ok=True)
paths=[]; solutions=[]
for p in sorted(root.glob('*-paths.csv')):
    paths.extend(csv.DictReader(p.open()))
for p in sorted(root.glob('*-solutions.csv')):
    solutions.extend(csv.DictReader(p.open()))
sol=defaultdict(list)
for s in solutions: sol[tuple(s[k] for k in ('case','seed','rep','method'))].append(s)
groups=defaultdict(list)
for p in paths: groups[(p['case'],p['method'])].append(p)
summary=[]; failures=[]
for (case,method),rs in sorted(groups.items()):
    valid=[]; maxgaps=[]; maxkkts=[]; statuses=defaultdict(int)
    smooth=method.startswith(('apg','newton'))
    for r in rs:
        statuses[r['status']]+=1
        ss=sol[tuple(r[k] for k in ('case','seed','rep','method'))]
        gaps=[float(s['gap']) for s in ss]
        kkts=[float(s['smooth_kkt']) for s in ss]
        maxgaps+=gaps
        maxkkts += [x for x in kkts if math.isfinite(x)]
        numeric=bool(ss) and all(math.isfinite(float(s['primal'])) and float(s['gap'])>=-1e-7 and float(s['sumdual'])<1e-10 and (float(s['smooth_kkt'])<=2e-5 if smooth else float(s['gap'])<=1e-5*(1+abs(float(s['primal'])))) for s in ss)
        complete=len(ss)==8 and int(r['levels'])==8
        if complete and numeric:
            valid.append(r)
        else:
            failures.append(dict(case=case,method=method,seed=r['seed'],rep=r['rep'],status=r['status'],levels=r['levels'],numeric=numeric,max_gap=max(gaps,default=None),max_kkt=max((x for x in kkts if math.isfinite(x)),default=None)))
    times=[float(r['total_s']) for r in valid]
    proc=defaultdict(list)
    for r in valid:proc[r['seed']].append(float(r['total_s']))
    means=[st.mean(v) for v in proc.values()]
    mu=st.mean(means) if means else math.nan
    sd=st.stdev(means) if len(means)>1 else math.nan
    t={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(len(means),2.0)
    half=t*sd/math.sqrt(len(means)) if len(means)>1 else math.nan
    summary.append(dict(case=case,method=method,valid_paths=len(valid),total_paths=len(rs),independent_inputs=len(means),mean_s=mu,sd_input_mean_s=sd,t95_lo=mu-half,t95_hi=mu+half,median_sample_s=st.median(times) if times else math.nan,max_gap=max(maxgaps,default=math.nan),max_smooth_kkt=max(maxkkts,default=math.nan),mean_setup_s=st.mean(float(r['setup_s']) for r in valid) if valid else math.nan,mean_optimize_s=st.mean(float(r['optimize_s']) for r in valid) if valid else math.nan,mean_bytes=st.mean(float(r['bytes']) for r in valid) if valid else math.nan,statuses=dict(statuses),hosts=sorted(set(r['host'] for r in rs))))
with (dest/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(dest/'failures.json').write_text(json.dumps(failures,indent=2))
(dest/'coverage.json').write_text(json.dumps(dict(paths=len(paths),solutions=len(solutions),groups=len(summary),invalid_or_incomplete=len(failures),files=len(list(root.glob('*-paths.csv')))),indent=2))
with (dest/'TABLE.md').open('w') as f:
    f.write('|Case|Method|Valid / attempted|Mean seconds|Input SD|95% t interval|Max exact gap|Max smooth KKT|\n|---|---|---:|---:|---:|---|---:|---:|\n')
    for s in summary:
        f.write(f"|{s['case']}|{s['method']}|{s['valid_paths']}/{s['total_paths']}|{s['mean_s']:.4g}|{s['sd_input_mean_s']:.3g}|[{s['t95_lo']:.3g}, {s['t95_hi']:.3g}]|{s['max_gap']:.3g}|{s['max_smooth_kkt']:.3g}|\n")
print((dest/'coverage.json').read_text())
