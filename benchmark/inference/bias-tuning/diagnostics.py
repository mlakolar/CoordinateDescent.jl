"""Summarize saved coefficient diagnostics on a Discovery compute node."""
from pathlib import Path
from collections import defaultdict
import csv,json,math
import numpy as np
groups=defaultdict(list)
with open('validation-holdout-coefficients.csv') as f:
 for row in csv.DictReader(f):groups[row['case'],row['method']].append(row)
out=[]
for (case,method),rows in sorted(groups.items()):
 active=[r for r in rows if float(r['truth'])!=0]
 z=np.array([(float(r['estimate'])-float(r['truth']))*np.sign(float(r['truth']))/float(r['se']) for r in active])
 fixed=[]
 if not case.endswith('random'):
  for target in sorted(set(r['target'] for r in active)):
   rr=[r for r in active if r['target']==target]
   e=np.array([float(r['estimate'])-float(r['truth']) for r in rr]);se=np.array([float(r['se']) for r in rr])
   fixed.append(float(e.std(ddof=1)/np.sqrt(np.mean(se**2))))
 d=dict(case=case,method=method,coefficients=len(rows),unmet_original_bias_target=sum(r['bias_target_met']!='true' for r in rows),grid_boundary=sum(r['path_boundary']=='true' for r in rows),selected_lambda_min=min(float(r['node_lambda']) for r in rows),selected_lambda_max=max(float(r['node_lambda']) for r in rows),active_signed_z_mean=float(z.mean()),active_signed_z_sd=float(z.std(ddof=1)),fixed_active_empirical_sd_over_rms_se=float(np.mean(fixed)) if fixed else None)
 out.append(d)
Path('validation-holdout-diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
# Retain the exact compact validation checks rather than classifying completed
# Slurm jobs as valid based only on scheduler exit status.
p=Path('final-report/validation-holdout-diagnostics.json');p.write_text(json.dumps(out,indent=2)+'\n')
