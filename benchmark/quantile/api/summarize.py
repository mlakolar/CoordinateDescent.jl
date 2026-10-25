"""Saved-output numerical summaries; run under Slurm with the fits."""
import csv,json,math,statistics as st,sys
from pathlib import Path
root=Path(sys.argv[1]);out=root/'summary';out.mkdir(exist_ok=True)
paths=[r for f in (root/'results').glob('*-paths.csv') for r in csv.DictReader(f.open())]
sol=[r for f in (root/'results').glob('*-solutions.csv') for r in csv.DictReader(f.open())]
coeff=[r for f in (root/'results').glob('*-coefficients.csv') for r in csv.DictReader(f.open())]
groups={}
for r in paths:groups.setdefault((r['case'],r['method']),[]).append(r)
summaries=[]
for (case,method),rows in sorted(groups.items()):
 valid=[r for r in rows if r['qualified']=='true' and r['raw_dual_available']=='true' and math.isfinite(float(r['total_s']))]
 pm={}
 for r in valid:pm.setdefault(r['seed'],[]).append(float(r['total_s']))
 ms=[st.mean(x) for x in pm.values()];mu=st.mean(ms) if ms else None;sd=st.stdev(ms) if len(ms)>1 else None
 t={2:12.706,3:4.303}.get(len(ms),None);half=t*sd/len(ms)**.5 if t else None
 cr=[s for s in coeff if s['case']==case and s['method']==method];cm={}
 for s in cr:cm.setdefault((s['seed'],s['rep']),[]).append(s)
 coeffok=sum(len(rr)==int(rows[0]['requested_levels']) and all(-1e-7<=float(s['check_gap'])<=2e-5*(1+abs(float(s['check_objective']))) for s in rr) for rr in cm.values())
 stock=[float(r['fit_s']) for r in rows if math.isfinite(float(r['fit_s']))]
 cp={}
 for r in rows:
  if math.isfinite(float(r['fit_s'])):cp.setdefault(r['seed'],[]).append(float(r['fit_s']))
 cs=[st.mean(v) for v in cp.values()]
 summaries.append(dict(case=case,method=method,attempted=len(rows),qualified_primal_dual=len(valid),independent_inputs=len(ms),mean_total_s=mu,input_sd_s=sd,t95_lo=mu-half if half is not None else None,t95_hi=mu+half if half is not None else None,
                       coefficient_paths_returned=len(cm),coefficient_paths_exact_gap_pass=coeffok,coefficient_only_fit_mean_s=st.mean(cs) if cs else None,all_observations_present=len(rows)==6))
(out/'summary.json').write_text(json.dumps(summaries,indent=2))
with (out/'TABLE.md').open('w') as f:
 f.write('|Case|Method|Qualified primal–dual /6|Mean full-output s|Input SD|95% interval|Coefficient-only returned / gap-pass|\n|---|---|---:|---:|---:|---|---|\n')
 def fmt(v):return '—' if v is None else f'{v:.4g}'
 for r in summaries:
  f.write(f"|{r['case']}|{r['method']}|{r['qualified_primal_dual']}/6|{fmt(r['mean_total_s'])}|{fmt(r['input_sd_s'])}|[{fmt(r['t95_lo'])}, {fmt(r['t95_hi'])}]|{r['coefficient_paths_returned']}/6; {r['coefficient_paths_exact_gap_pass']}/6|\n")
coverage=dict(path_rows=len(paths),expected_path_rows=240,groups=len(groups),expected_groups=40,input_files=len(list((root/'results').glob('*-paths.csv'))),expected_input_files=15,solution_rows=len(sol),coefficient_rows=len(coeff),complete=len(paths)==240 and len(groups)==40 and all(r['all_observations_present'] for r in summaries))
(out/'coverage.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage,indent=2))
contrasts=[]
for case in sorted({r['case'] for r in paths}):
 for baseline,candidate in [('gurobi-primal','api-exact'),('quantreg-native-dual','api-exact'),('conquer-dual','api-smooth')]:
  rr=[r for r in summaries if r['case']==case and r['method'] in [baseline,candidate]]
  if len(rr)!=2 or any(r['qualified_primal_dual']!=6 for r in rr):continue
  times={}
  for r in paths:
   if r['case']==case and r['method'] in [baseline,candidate]:times.setdefault((r['seed'],r['method']),[]).append(float(r['total_s']))
  ds=[st.mean(times[(seed,baseline)])-st.mean(times[(seed,candidate)]) for seed in sorted({k[0] for k in times})]
  sd=st.stdev(ds);half=4.303*sd/3**.5
  contrasts.append(dict(case=case,baseline=baseline,candidate=candidate,mean_saved_s=st.mean(ds),t95_lo=st.mean(ds)-half,t95_hi=st.mean(ds)+half))
(out/'paired-contrasts.json').write_text(json.dumps(contrasts,indent=2))
# Strong-duality and feasibility of the native quantreg workspace is an explicit
# qualification, not assumed merely because the wrapper returned a vector.
(out/'dual-checks.json').write_text(json.dumps({m:{'rows':len([r for r in sol if r['method']==m]),'max_dual_feasibility':max((float(r['dual_feasibility']) for r in sol if r['method']==m),default=None),'max_kkt':max((float(r['kkt']) for r in sol if r['method']==m),default=None)} for m in ['api-exact','api-smooth','gurobi-primal','quantreg-native-dual','conquer-dual']},indent=2))
