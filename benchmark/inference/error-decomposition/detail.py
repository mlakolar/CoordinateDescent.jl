"""Additional diagnostics from saved target records; execute on Discovery compute."""
import csv,json,math,statistics
from pathlib import Path
from collections import defaultdict
root=Path('reports/main');rows=list(csv.DictReader((root/'coefficients.csv').open()))
q=1.959963984540054
mean=statistics.mean
def mc(v):return statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
def quant(v,p):
 v=sorted(v);i=(len(v)-1)*p;lo=int(i);return v[lo]+(i-lo)*(v[min(lo+1,len(v)-1)]-v[lo])
checks={'finite_estimates_and_components':all(all(math.isfinite(float(r[k])) for k in ['estimate','se','true_variance_se','robust_se','stochastic','linear_bias','nonlinear_remainder']) for r in rows),'gaussian_nonlinear_max':max(abs(float(r['nonlinear_remainder'])) for r in rows if r['case'].startswith('gaussian')),'oracle_initial_linear_max':max(abs(float(r['linear_bias'])) for r in rows if r['initial']=='oracle'),'oracle_initial_nonlinear_max':max(abs(float(r['nonlinear_remainder'])) for r in rows if r['initial']=='oracle'),'exact_projection_linear_max':max(abs(float(r['linear_bias'])) for r in rows if r['profile']=='exact'),'max_noise_budget_ratio':max(float(r['noise_ratio']) for r in rows)}
index={(r['case'],r['rep'],r['profile'],r['weights'],r['initial'],r['target']):r for r in rows}
groups=defaultdict(lambda:defaultdict(list));targetgroups=defaultdict(list)
ordinals=defaultdict(int)
for r in rows:
 key=(r['case'],r['profile'],r['weights'],r['initial']);role='active' if float(r['truth']) else 'null';groups[key+(role,)][r['rep']].append(r)
 okey=key+(r['rep'],role);ordinals[okey]+=1
 label=role+'-slot-'+str(ordinals[okey]) if r['case'].endswith('random') else 'target-'+r['target']
 targetgroups[key+(label,)].append(r)
def metrics(rr):
 def signed(r):return 1 if float(r['truth'])>=0 else -1
 result={'true_variance_coverage':mean([float(abs(float(r['estimate'])-float(r['truth']))<=q*float(r['true_variance_se'])) for r in rr]),'robust_coverage':mean([float(abs(float(r['estimate'])-float(r['truth']))<=q*float(r['robust_se'])) for r in rr]),'absolute_linear_z':mean([abs(float(r['linear_bias'])/float(r['se'])) for r in rr]),'absolute_nonlinear_z':mean([abs(float(r['nonlinear_remainder'])/float(r['se'])) for r in rr])}
 base=[index[(r['case'],r['rep'],'base',r['weights'],r['initial'],r['target'])] for r in rr]
 result['absolute_estimate_change_over_base_se']=mean([abs(float(r['estimate'])-float(b['estimate']))/float(b['se']) for r,b in zip(rr,base)])
 result['absolute_se_fractional_change']=mean([abs(float(r['se'])/float(b['se'])-1) for r,b in zip(rr,base)])
 return result
summary=[]
for key,reps in sorted(groups.items()):
 d=dict(zip(['case','profile','weights','initial','target_class'],key));d['datasets']=len(reps)
 per=[metrics(rr) for rr in reps.values()]
 for name in per[0]:v=[x[name] for x in per];d[name]=mean(v);d[name+'_mcse']=mc(v)
 rr=[r for rep in reps.values() for r in rep]
 for name in ['stochastic','linear_bias','nonlinear_remainder']:
  v=[float(r[name])/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr]
  d[name+'_z_sd']=statistics.stdev(v);d[name+'_z_q025']=quant(v,.025);d[name+'_z_q975']=quant(v,.975)
 summary.append(d)
targetsummary=[]
for key,rr in sorted(targetgroups.items()):
 d=dict(zip(['case','profile','weights','initial','target'],key));d['datasets']=len(rr)
 for name in ['covered','rejected']:
  v=[float(r[name]=='true') for r in rr];d[name]=mean(v);d[name+'_mcse']=mc(v)
 for name in ['stochastic','linear_bias','nonlinear_remainder']:
  v=[float(r[name])/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr];d[name+'_signed_z']=mean(v);d[name+'_signed_z_mcse']=mc(v)
 targetsummary.append(d)
resources=[]
for p in sorted(Path('resources').glob('main-*.txt')):
 d=dict(line.split('=',1) for line in p.read_text().splitlines());d['file']=p.name;resources.append(d)
checks['worker_resource_records']=len(resources);checks['worker_exits_zero']=all(r['process_exit']=='0' for r in resources)
(root/'detail-checks.json').write_text(json.dumps(checks,indent=2));(root/'detail.json').write_text(json.dumps(summary,indent=2));(root/'targets.json').write_text(json.dumps(targetsummary,indent=2));(root/'resources.json').write_text(json.dumps(resources,indent=2))
base=json.loads((root/'summary.json').read_text());pairs=json.loads((root/'paired.json').read_text())
lines=['# Discovery error decomposition and projection-path stability','', 'EMPIRICAL. Frozen experimental restricted LDPE m4/kappa.5; production defaults unchanged. Main:200 datasets per four logistic cases and100 per two Gaussian controls. Oracle interventions are diagnostics only. All uncertainties below are dataset-level Monte Carlo standard errors.','', '## Logistic active coverage and signed error decomposition','', '| Case | Weights | Initial | Coverage +/- MCSE | Stochastic z | Linear z | Nonlinear z |','|---|---|---|---|---|---|---|']
for r in base:
 if r['case'].startswith('binomial') and r['profile']=='base' and r['target_class']=='active':
  lines.append(f"| {r['case']} | {r['weights']} | {r['initial']} | {100*r['coverage']:.1f}% +/- {100*r['coverage_mcse']:.1f}pp | {r['stochastic_signed_z']:.3f} | {r['linear_bias_signed_z']:.3f} | {r['nonlinear_remainder_signed_z']:.3f} |")
lines+=['','## Grid effects with estimated initialization and weights','','| Case | Grid | Active coverage +/- MCSE | Null type I | Active interval length | Lower endpoint |','|---|---|---|---|---|---|']
for r in base:
 if r['weights']=='estimated' and r['initial']=='estimated' and r['target_class']=='active':
  null=next(v for v in base if all(v[k]==r[k] for k in ['case','profile','weights','initial']) and v['target_class']=='null')
  lines.append(f"| {r['case']} | {r['profile']} | {100*r['coverage']:.1f}% +/- {100*r['coverage_mcse']:.1f}pp | {100*null['rejection']:.1f}% | {r['interval_length']:.4f} | {100*r['lower_endpoint_fraction']:.1f}% |")
lines+=['','## Paired grid changes','','| Case | Grid | Absolute estimate change / base SE | Absolute fractional SE change | Coverage change +/- MCSE |','|---|---|---|---|---|']
for r in summary:
 if r['weights']=='estimated' and r['initial']=='estimated' and r['target_class']=='active' and r['profile']!='base':
  pair=next(v for v in pairs if all(v[k]==r[k] for k in ['case','profile','weights','initial','target_class']))
  lines.append(f"| {r['case']} | {r['profile']} | {r['absolute_estimate_change_over_base_se']:.5f} | {r['absolute_se_fractional_change']:.5f} | {100*pair['coverage_difference']:.2f} +/- {100*pair['coverage_difference_mcse']:.2f}pp |")
lines+=['','## Checks and limitations','', '```json',json.dumps(checks,indent=2),'```','', 'Signed components are aligned with the true coefficient sign for active targets. Component quantiles/SD and per-target summaries are in detail.json and targets.json; targets within datasets are dependent. Random-support targets are reported by selection slot rather than pooling unrelated coordinate IDs. The true-variance quadratic form is diagnostic when scores depend on the response. Gaussian controls use known sigma1. Exact inverse projection is restricted to benign p<n. No Poisson, competitor, or simultaneous-coverage validation is included. Grids change discrete stage-1 reference as well as floor and density. Raw score vectors were not retained; paired stability metrics describe estimates and SE, not a norm of score-vector changes. Timing includes compilation and reuse and is not an end-to-end competitor benchmark.']
if resources:lines+=['',f"Worker processes: {len(resources)}; maximum whole-worker RSS {max(float(r['max_rss_kib']) for r in resources)/1024:.1f} MiB. Process seconds sum {sum(float(r['process_seconds']) for r in resources):.1f}; CPU seconds sum {sum(float(r['cpu_seconds']) for r in resources):.1f}. These include many diagnostic interventions per dataset."]
(root/'RESULTS.generated.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(checks,indent=2))
if not checks['finite_estimates_and_components'] or checks['gaussian_nonlinear_max']>1e-8 or checks['oracle_initial_linear_max']>1e-8 or checks['oracle_initial_nonlinear_max']>1e-8 or checks['exact_projection_linear_max']>1e-8 or checks['max_noise_budget_ratio']>1.50001 or len(resources)!=48 or not checks['worker_exits_zero']:raise SystemExit(1)
