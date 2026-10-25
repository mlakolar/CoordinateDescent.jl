"""Aggregate saved convergence measurements inside their Slurm allocation."""
import csv,json,math,statistics as st,re
from pathlib import Path
root=Path('.');out=root/'summary';out.mkdir(exist_ok=True)
rows=[r for f in (root/'results').glob('*-paths.csv') for r in csv.DictReader(f.open())]
groups={}
for r in rows:groups.setdefault((r['case'],r['method'],r['seed']=='1202'),[]).append(r)
summary=[]
for (case,method,anchor),rr in sorted(groups.items()):
 valid=[r for r in rr if r['qualified_2e6']=='true' and math.isfinite(float(r['total_s']))]
 per={}
 for r in valid:per.setdefault(r['seed'],[]).append(float(r['total_s']))
 means=[st.mean(v) for v in per.values()];mu=st.mean(means) if means else None;sd=st.stdev(means) if len(means)>1 else None
 half=4.303*sd/3**.5 if len(means)==3 else None
 summary.append(dict(case=case,method=method,anchor=anchor,attempts=len(rr),qualified_2e6=len(valid),qualified_2e7=sum(r['qualified_2e7']=='true' for r in rr),inputs=len(per),mean_total_s=mu,input_sd_s=sd,t95_lo=mu-half if half is not None else None,t95_hi=mu+half if half is not None else None,max_kkt=max((float(r['max_kkt']) for r in rr if math.isfinite(float(r['max_kkt']))),default=None),max_certificate_gap=max((float(r['max_certificate_gap']) for r in rr if math.isfinite(float(r['max_certificate_gap']))),default=None)))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
with (out/'TABLE.md').open('w') as f:
 f.write('|Case|Method|Anchor|Qualified / attempts|Mean s|Input SD|95% interval|Max KKT|\n|---|---|---|---:|---:|---:|---|---:|\n')
 fmt=lambda v:'—' if v is None else f'{v:.4g}'
 for r in summary:f.write(f"|{r['case']}|{r['method']}|{r['anchor']}|{r['qualified_2e6']}/{r['attempts']}|{fmt(r['mean_total_s'])}|{fmt(r['input_sd_s'])}|[{fmt(r['t95_lo'])}, {fmt(r['t95_hi'])}]|{fmt(r['max_kkt'])}|\n")
contrasts=[]
for case in ("small","tail","tall","wide"):
 for candidate in ("certified","certified-tight"):
  baseline="conquer-1e-7"
  rr=[r for r in rows if r['case']==case and r['seed']!='1202' and r['method'] in (baseline,candidate)]
  if len(rr)!=12 or any(r['qualified_2e6']!='true' for r in rr):continue
  per={}
  for r in rr:per.setdefault((r['seed'],r['method']),[]).append(float(r['total_s']))
  ds=[st.mean(per[(seed,baseline)])-st.mean(per[(seed,candidate)]) for seed in ("1301","1302","1303")]
  mu=st.mean(ds);half=4.303*st.stdev(ds)/3**.5
  contrasts.append(dict(case=case,candidate=candidate,baseline=baseline,mean_saved_s=mu,t95_lo=mu-half,t95_hi=mu+half))
(out/'paired-contrasts.json').write_text(json.dumps(contrasts,indent=2))
profiles=[]
for f in (root/'results').glob('*profile.txt'):
 frames=[]
 for line in f.read_text().splitlines():
  m=re.match(r'\s*(\d+)\s+(\d+)\s+(.+)',line)
  if m and int(m[2]):frames.append(dict(inclusive=int(m[1]),self=int(m[2]),frame=m[3]))
 profiles.append(dict(file=f.name,top_self=sorted(frames,key=lambda x:x['self'],reverse=True)[:12]))
(out/'profiles.json').write_text(json.dumps(profiles,indent=2))
coverage=dict(path_rows=len(rows),expected_rows=130,groups=len(groups),expected_groups=25,input_files=len(list((root/'results').glob('*-paths.csv'))),expected_input_files=13,complete=len(rows)==130 and len(groups)==25)
(out/'coverage.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage,indent=2))
