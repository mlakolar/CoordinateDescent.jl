"""Collect final reporting inputs and update conditional costs, on Discovery."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np
root=Path.cwd();out=root/'final-report'
coeff=[]
for f in (root/'default-refresh').glob('*/*.coefficients'):coeff.extend(csv.DictReader(f.open()))
assert len(coeff)==2000
with (root/'default-refresh-coefficients.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=coeff[0],lineterminator='\n');w.writeheader();w.writerows(coeff)
resources=[]
for f in (root/'resources').glob('refresh-v6-*.txt'):
 resources.append(dict(file=str(f.relative_to(root)),**{k:float(v) for k,v in (x.split('=') for x in f.read_text().split())}))
assert len(resources)==32 and all(r['process_exit']==0 for r in resources)
rows=list(csv.DictReader((out/'current-pilot-records.csv').open()))
groups=defaultdict(list)
for r in rows:
 if r['mode']!='experimental' and r['status']=='ok':groups[r['scenario'],r['engine'],r['mode']].append(r)
assert len(groups)==34 and all(len(rs)==20 for rs in groups.values())
warm=sum(np.mean([float(r['fit_seconds']) for r in rs])*1000/3600 for rs in groups.values())
receipt=dict(current_default_refresh_fits=200,current_default_refresh_coefficients=len(coeff),current_default_refresh_worker_processes=len(resources),current_default_refresh_max_rss_kib=max(r['max_rss_kib'] for r in resources),current_default_refresh_resources=resources,
 full_original_protocol=dict(cases=4,supported_cells=34,datasets_per_case=1000,warm_cpu_hours=float(warm),ideal_eight_worker_hours=float(warm/8),eight_worker_hours_with_25pct_margin=float(1.25*warm/8),scope='Original four-case/ten-requested-target protocol with current Julia default fits replacing historical rows; other cached fits and explicit matched penalties unchanged. Candidate excluded. Warm throughput projection only; process startup and queue delay excluded. Not an all-p or eight-design comparison estimate.'))
(out/'final-receipts.json').write_text(json.dumps(receipt,indent=2)+'\n')
report=(out/'RESULTS.md').read_text()
report=report.replace('All Monte Carlo standard errors below use dataset-level summaries, retaining dependence between targets.','All Monte Carlo standard errors below use dataset-level summaries, retaining dependence between targets. Replications are independent within each case. Corresponding Gaussian and binomial cases share their covariate draw;1,600 datasets includes both response families rather than1,600 distinct covariate matrices.')
report+='''

The refreshed QP defaults also fail calibration: active coverage36% and8%
in the moderate/correlated Gaussian paired20-dataset pilot, respectively.
Numerical feasibility and dual gaps do not establish statistical validity.

## Bias and grid diagnostics

Every candidate coefficient attained the original working-design bias-factor bound,
and every recorded noise ratio obeyed the frozen50% SD budget. These numerical
conditions did not establish coverage: logistic intervals still undercovered.
The selected penalty was a finite-grid endpoint for2,000/2,000 target rows in
both benign cases,1,641/2,000 in Gaussian moderate, and1,568/2,000 in logistic
moderate. This leaves a material grid-boundary limitation; it is not evidence
that an unconstrained continuous-path optimum was found. We did not extend the
grid or retune using holdout results.

The signed active error divided by SE averaged−.283 for Gaussian moderate and
−.724 for logistic moderate under the candidate. The corresponding empirical
SD/RMS-SE ratios were.968 and.943. Remaining logistic undercoverage therefore
cannot be repaired merely by claiming that its reported SEs are too small.
The benign logistic case had a ratio1.090 and signed bias−.121, a different
finite-sample calibration pattern. See validation-holdout-diagnostics.json.

## Current-default competitor refresh

CURRENT_COMPARISON.md updates the native Julia default rows on exactly the
original20shared datasets per case. All200refresh fits succeeded, including
the current nodewise defaults, Gaussian QP defaults, and the frozen experimental
candidate. It reuses the validated competitor and explicit-matched fits rather
than rerunning them. These legacy pilot datasets are separate from the200fresh
holdout datasets per case; the refresh does not select or retune the candidate.
Full pipelines still compute different internal target counts. The separate
500-slope/row timing benchmark remains the matched engine comparison.
'''
report+=f"\nFor the refreshed original34-cell/four-case protocol at1,000datasets/case, measured mean warm fitting costs project to{warm:.2f}CPU-hours: an ideal{warm/8:.2f}hours on8workers, or{1.25*warm/8:.2f}allocated hours with25%margin. This excludes process startup, queue delay and storage. Eight CPUs/16GiB remain the planning allocation. The historical fresh-process estimate was about50allocated hours; current Julia automatic-pipeline cold startup has not been measured separately per policy, so50hours is not a confirmed current full-run budget. Batching replications amortizes startup. No full comparison was launched.\n"
(out/'RESULTS.md').write_text(report)
print(json.dumps({k:v for k,v in receipt.items() if k!='current_default_refresh_resources'},indent=2))
