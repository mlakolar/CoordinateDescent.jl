from pathlib import Path
import json,csv,math
import numpy as np
root=Path.cwd();v=json.loads(Path('validation.json').read_text());b=json.loads(Path('benchmark-records.json').read_text());bridge=json.loads(Path('bridge-comparison.json').read_text());ss=list(csv.DictReader(open('study-summary.csv')))
assert v['automatic_datasets']==v['automatic_converged']==800 and v['benchmark_fits']==v['benchmark_valid']==140
assert not v['study_error_files']
for stage in ['development','holdout']:
 for case in ['gaussian-moderate','gaussian-correlated','binomial-moderate','binomial-correlated']:
  rs=[r for r in ss if r['stage']==stage and r['case']==case and r['method']=='automatic'];assert len(rs)==1 and int(rs[0]['datasets'])==100
parity={}
for comp in ['ours-hdi','ours-econml','oursqp-sslasso']:
 rs=[r for r in v['parity'] if r['comparison']==comp];assert len(rs)==20
 parity[comp]={key:max(r[key] for r in rs) for key in rs[0] if key not in ['case','rep','comparison']}
cert={}
for engine in ['oursqp','sslasso']:
 rs=[r for r in b if r['engine']==engine];cert[engine]={'max_abs_gap':max(float(r['qp_gap']) for r in rs),'max_residual':max(float(r['qp_residual']) for r in rs)}
bridge_summary={'datasets':len(bridge),'max_beta_difference':max(r['beta_max_abs'] for r in bridge),'max_se_difference':max(r['se_max_abs'] for r in bridge),'max_initial_lambda_difference':max(abs(r['lambda_old']-r['lambda_head']) for r in bridge),'max_node_lambda_difference':max(abs(r['node_old']-r['node_head']) for r in bridge)}
# Same measured workload, extended to 1000 datasets per Gaussian case. This
# includes QP components and is explicitly not the complete statistical study.
cost={}
for label in ['moderate','correlated']:
 rs=[r for r in b if r['case']==label];assert len(rs)==70
 cost[label]={k:sum(float(r[k]) for r in rs)*100/3600 for k in ['fit_seconds','process_seconds']}
subset_fit=sum(r['fit_seconds'] for r in cost.values());subset_process=sum(r['process_seconds'] for r in cost.values())
subset={'scope':'Exact seven timing tasks per dataset, 1000 datasets per Gaussian case; component and complete-pipeline tasks differ. Excludes native-default competitors and binomial cases; not the full comparison.','warmed_fit_cpu_hours':subset_fit,'fresh_process_cpu_hours':subset_process,'ideal_fresh_process_wall_hours_8cpu':subset_process/8,'planning_wall_hours_8cpu_25pct_margin':subset_process/8*1.25,'cpus':8,'memory_gib':16,'max_process_rss_mib':max(float(r['max_rss_kib'])/1024 for r in b)}
ratios=[]
for label in ['moderate','correlated']:
 for alt in ['hdi','econml']:
  vals=[]
  for rep in range(1,11):
   ours=next(r for r in b if r['engine']=='ours' and r['case']==label and int(r['rep'])==rep);other=next(r for r in b if r['engine']==alt and r['case']==label and int(r['rep'])==rep)
   vals.append(float(other['fit_seconds'])/float(ours['fit_seconds']))
  ratios.append(dict(case=label,alternative=alt,paired_median=float(np.median(vals)),minimum=min(vals),maximum=max(vals)))
final=dict(parity=parity,qp_certificates=cert,current_source_bridge=bridge_summary,timing_workload_extrapolation=subset,paired_warm_speed_ratios=ratios)
Path('final-validation.json').write_text(json.dumps(final,indent=2)+'\n')
lines=Path('RESULTS.md').read_text().splitlines()
lines += ['','## Interpretation','', '**Do not launch the full comparison yet.** Automatic prediction CV removes required manual tuning and adapts to correlation, but active-coefficient intervals remain substantially below 95% coverage, including the moderate designs. Numerical convergence does not resolve this calibration failure. The policy was frozen before seeing these results; it has not been retuned against the holdout.','', '## Bias and noise diagnostics','', '| Holdout Gaussian case | Estimated-noise coverage | Known-noise coverage | Oracle initializer + known noise coverage | Signed bias / SE | Empirical SD / RMS SE |','|---|---:|---:|---:|---:|---:|']
for case in ['gaussian-moderate','gaussian-correlated']:
 rs={r['method']:r for r in ss if r['stage']=='holdout' and r['case']==case}
 a=rs['automatic'];known=rs['known_sigma'];oracle=rs['oracle_initial_and_sigma']
 lines.append(f"| {case} | {float(a['coverage_active']):.3f} | {float(known['coverage_active']):.3f} | {float(oracle['coverage_active']):.3f} | {float(a['signed_z_bias_active']):.3f} | {float(a['empirical_sd_over_rms_se_active']):.3f} |")
lines += ['','Oracle calculations use truth and are diagnostics, not feasible competing methods. Noise estimation alone does not explain the observed failures. Prediction-optimal penalization leaves appreciable inferential bias.','', '## Numerical comparison and current source','', '```json',json.dumps(dict(parity=parity,qp_certificates=cert,current_source_bridge=bridge_summary),indent=2),'```','', 'EconML divides residual sum of squares by n minus the number of selected slopes; the Julia/hdi adapter subtracts an additional intercept degree of freedom. The nearly constant SE ratio is therefore expected. All QP rows satisfy the declared feasibility and optimality tolerances. The authors reference includes the documented diagonal fast-path correction.','', 'Measured source is commit f7b587790f46b6ea4c05f082971d582db0ddb435 with the original pinned solver dependencies. After merging master, inference CV explicitly retains gaptol=0 unless the user overrides it. The current-source bridge uses ProximalBase b6c949f and ProximalSolvers 73a1c8e; all 115 inference/golden/tuning checks passed, and three independent datasets per case were compared with the saved holdout fits.','', '## Runtime and resource interpretation','', 'The Julia fit kernels are faster under equivalent fixed tuning. Fresh Julia process startup/JIT dominates these small problems: measured median complete process times are roughly 13–18 seconds versus roughly 3–4 seconds for hdi/EconML. Persistent workers would change this comparison and were not timed here. Peak Julia process RSS is about 0.8 GiB versus about 0.25 GiB for hdi/EconML; eight workers fit within the 16 GiB allocation.','', 'The exact timing workload extrapolated to 1000 datasets per Gaussian case gives the following conditional estimate. It includes component-only QP tasks and the separate ten-target pipelines; it is not an estimate for the complete four-case defaults-and-matched inference comparison.','', '```json',json.dumps(subset,indent=2),'```','', 'The earlier full-comparison planning budget was about 50 allocated wall-hours at eight CPUs and 16 GiB for the original ten-target task. That remains a provisional reference, not a refreshed validated budget for revised defaults or different target counts. Freeze an inferentially defensible automatic rule and the final comparison protocol before requesting a firm full-study budget. Queue waiting is excluded.','', '## Validation and recovery','', 'All 800 automatic fits converged and all 140 timing fits produced valid outputs. The first benchmark allocation hit its eight-minute wall limit with two sslasso fits unfinished; only those two were recovered. Successful fits were reused and interrupted logs retained. The 115-check source canary and current-source canary both passed. Full GitHub correctness CI and documentation passed on the preceding merged source; the final explicit CV stopping-rule change is also covered by the current-source Discovery tests. The full 1000-replication comparison was not submitted.']
Path('RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(final,indent=2))
