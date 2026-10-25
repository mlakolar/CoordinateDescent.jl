"""Build the final frozen-study report on a Discovery compute node."""
from pathlib import Path
import csv,json,shutil
root=Path.cwd();out=root/'final-report';out.mkdir(exist_ok=True)
validation=json.loads((root/'validation-holdout-summary.json').read_text())
checks=json.loads((root/'validation-holdout-checks.json').read_text())
development=json.loads((root/'restricted-development-summary.json').read_text())
assert len(validation['summary'])==16 and all(d['records']==d['converged']==200 for d in validation['summary'])
assert not validation['errors'] and checks['checks']['records']==3200
assert len(development['summary'])==40 and all(d['records']==d['converged']==100 for d in development['summary'])
assert not development['errors']
assert checks['checks']['resources']['worker_processes']==64
files=['validation-holdout-summary.json','validation-holdout-checks.json','validation-holdout-records.csv',
 'restricted-development-summary.json','restricted-development-records.csv','development-summary.json','development-records.csv']
for name in files:shutil.copyfile(root/name,out/name)
# Retain full coefficients at the run root and in the local run directory. The
# compact public report includes per-dataset metrics needed to recreate tables.
rows=validation['summary']
lines=['# Bias-aware tuning: frozen fresh-data validation','',
 'EMPIRICAL. This follow-up tested one frozen experimental policy against the unchanged CV default. '
 'The full 1,000-replication cross-method comparison was not submitted.','',
 'Development used 100 datasets in each of eight designs (800 datasets, 4,000 converged fits). '
 'After development, restricted LDPE with four unpenalized correlations and a 50% noise-SD budget '
 'was frozen before using the independent holdout namespace. Fresh validation used 200 datasets '
 'per design (1,600 datasets, 3,200 fits). Five active and five null targets belong to each dataset. '
 'All Monte Carlo standard errors below use dataset-level summaries, retaining dependence between targets.','',
 '| Case | Policy | Active coverage ± MCSE | Null coverage | Type I | Power | Length active/null | Median warm automatic seconds |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
for d in rows:
 lines.append(f"| {d['case']} | {d['method']} | {100*d['coverage_active']:.1f}% ± {100*d['coverage_active_mcse']:.1f} pp | {100*d['coverage_null']:.1f}% | {100*d['type1']:.1f}% | {100*d['power']:.1f}% | {d['length_active']:.3f}/{d['length_null']:.3f} | {d['seconds_median']:.3f} |")
lines+=['','The nominal marginal targets are 95% coverage and 5% type I error. '
 'Paired differences and Monte Carlo errors for every reported quantity are in validation-holdout-checks.json. '
 'Mean and median timing summaries and Monte Carlo errors for coverage, error, power and interval length '
 'are in validation-holdout-summary.json. These are empirical checks, not finite-sample guarantees.','',
 '## Calibration decision','']
problem=[]
for d in rows:
 if d['method']=='rldpe4_k50' and d['coverage_active']+1.96*d['coverage_active_mcse']<.95:problem.append(d['case'])
if problem:
 lines.append('Do not promote the candidate as a calibrated general-purpose default. '
 'The approximate 95% Monte Carlo upper limit for active coverage is below 95% in: '+', '.join(problem)+'. '
 'These limits quantify simulation error only; they do not repair inferential bias. '
 'The policy is not retuned against these validation datasets.')
else:
 lines.append('No case has an approximate 95% Monte Carlo upper limit below nominal active coverage. '
 'This is not proof of calibration or equivalence. Assess the reported uncertainty, null errors, interval length and power before promotion.')
lines+=['','## Runtime, memory and comparison scope','']
for policy in ['cv','rldpe4_k50']:
 cpu_hours=sum(d['seconds']*1000/3600 for d in rows if d['method']==policy)
 lines.append(f"Conditional own-policy projection for all eight designs, 1,000 datasets/design and ten targets: {policy} needs about {cpu_hours:.2f} warm CPU-hours, or {1.25*cpu_hours/8:.2f} allocated wall-hours on eight single-CPU workers with a 25% margin. This multiplies measured mean complete fit costs; it excludes competitor fits, process startup, scheduling and storage. It is not a full comparison estimate.")
lines.append('')
res=checks['checks']['resources']
lines.append(f"The fresh validation workers used one CPU each. Across {res['worker_processes']} worker processes, the maximum measured peak RSS was {res['maximum_rss_kib']/1024:.1f} MiB. Each worker executed both policies on a dataset shard; this is a combined process peak, not a separate per-method memory estimate. The allocation was eight CPUs and 16 GiB. Warm fit timings include each complete automatic policy, including its own initial CV. Process wall time, CPU usage and peak RSS are retained in the worker resource records.")
lines+=['',
 'The earlier matched 20-dataset competitor speed benchmark remains the fixed-penalty engine comparison. '
 'It has 140 valid fits: Julia/hdi/EconML use all 500 slopes, Julia/sslasso all 500 QP rows, '
 'and SIHR is reported separately as a different ten-target pipeline. The new policy changes '
 'projection tuning and interval width, so the earlier engine timings are not its automatic end-to-end speedups. '
 'Default CV fitting is unchanged apart from rejecting unusable CV tail entries; the package rename '
 'and current master dependency bridge reproduced 120 fits on 24 datasets exactly.','',
 'The earlier approximately 14 allocated wall-hour estimate concerns only the seven-task Gaussian '
 'matched timing workload, on eight CPUs/16 GiB with a 25% margin. It excludes native competitor '
 'defaults and binomial cases. The earlier approximately 50-hour complete-protocol estimate covers the four original pilot designs and is '
 'provisional; a final cross-method inferential policy and protocol have not been validated or frozen. '
 'No full-study launch is justified by these runtime estimates alone.','',
 '## Provenance and limitations','',
 '- Numerical source: immutable source-v4/env-v4, based on commit1c6e35b. Frozen runner recipe: f1b73f8. '
 'The registry records full hashes, dependency commits, job IDs and resource requests.',
 '- Four initial cases use n=200,p=500,rho=.3,s=5 or n=100,p=500,rho=.9,s=15. '
 'Benign cases use n=500,p=100,rho=.3,s=5; random-support cases use the moderate dimensions. '
 'Gaussian signals alternate ±.5 and binomial signals ±.7. The intercept is .2 and Gaussian noise SD is one.',
 '- The initializer uses five-fold 40-level prediction CV with opt_tol1e-8 and inference gaptol0. '
 'The candidate uses a 60-level LDPE path to .001 of the conditional maximum; m=4, kappa=.5, '
 'bias bound sqrt(2log(p)), relaxation .1. Model-based marginal normal intervals are evaluated.',
 '- Development, holdout and earlier speed datasets use separate seed namespaces. '
 'The published restricted-projection rule motivates the candidate; the GLM Hessian application '
 'is evaluated empirically and has no new theorem here.',
 '- The first development probe tested selected-model refits and halved initial penalties. '
 'Neither repaired coverage consistently. Its adverse results remain in development-summary.json.',
 '- The CV tail warm-up failure and R library-path failure were recovered without rerunning valid Gaussian results. '
 'The first holdout resource wrapper failed before any Julia fitting because /usr/bin/time was unavailable; '
 'its five-second failure and empty-data analysis are retained. A tested Python resource wrapper recovered '
 'the untouched study with the same statistical source, runner and seeds.',
 '- Independent glmnet checks passed for ordinary and restricted LDPE, and 142 inference/canary checks passed. '
 'The current package/master-dependency bridge passed 9,624 checks with zero maximum numerical difference. '
 'Hosted CI and docs passed on the integrated implementation before the recipe-only follow-up.',
 '', 'See the preceding automatic-tuning report for native competitor coverage, matched speed, '
 'EconML intercept degrees-of-freedom scaling, and QP feasibility/optimality certificates. '
 'Full coefficient records, raw worker logs, failures and accounting are retained in the task run folder.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
