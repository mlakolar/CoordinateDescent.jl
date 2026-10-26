"""Compute-node-only numerical summary of native RCAL versus experimental Julia loss."""
import csv,json,math,statistics,sys
from pathlib import Path
mode=sys.argv[1];folder='calibration-engine' if mode=='fixed' else 'calibration-cv-engine'
root=Path('outputs')/folder;dest=Path('reports')/folder;dest.mkdir(parents=True,exist_ok=True)
cases=['binomial-moderate','binomial-benign','binomial-largen','liu-published'];rows=[];errors=[];warnings=[];missing=[]
filename='julia-checks.csv' if mode=='fixed' else 'julia-cv-checks.csv'
for case in cases:
 for rep in range(1,6):
  d=root/case/str(rep);f=d/filename
  if f.exists():rows.extend(csv.DictReader(f.open()))
  else:missing.append([case,rep])
  errors.extend(dict(file=str(p),error=p.read_text()) for p in d.glob('*.error'))
  warnings.extend(dict(file=str(p),warning=p.read_text()) for p in d.glob('*.warnings'))
if rows:
 with (dest/'records.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finite_numbers(key):return [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
checks=dict(scope=mode,planned_datasets=20,planned_policy_rows=40,rows=len(rows),missing_dataset_reports=missing,errors=len(errors),warning_files=len(warnings),all_converged=all(r['converged'].lower()=='true' for r in rows),max_julia_kkt=max(finite_numbers('kkt'),default=None),max_reference_kkt=max(finite_numbers('reference_kkt'),default=None))
if mode=='fixed':
 checks.update(parity_rows_passed=sum(r['parity_pass'].lower()=='true' for r in rows),max_coefficient_error=max(finite_numbers('max_coefficient_error'),default=None),max_predictor_error=max(finite_numbers('max_predictor_error'),default=None),max_objective_error=max((abs(float(r['objective'])-float(r['reference_objective'])) for r in rows),default=None))
else:
 checks.update(matched_selected_index=sum(float(r['index'])==float(r['reference_index']) for r in rows),reference_index_rows=len(finite_numbers('reference_index')),max_coefficient_error=max(finite_numbers('coefficient_error'),default=None),max_predictor_error=max(finite_numbers('predictor_error'),default=None),max_objective_error=max(finite_numbers('objective_error'),default=None),max_cv_loss_error=max(finite_numbers('cv_max_loss_error'),default=None))
for name,obj in [('checks',checks),('errors',errors),('warnings',warnings)]:(dest/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
lines=['# Experimental calibrated-loss solver checks','','EMPIRICAL. Native RCAL versus a custom ProximalBase ScalarLoss through the existing Julia GLM proximal-Newton solver. This is not an inference coverage experiment and does not promote a logistic default.','','```json',json.dumps(checks,indent=2),'```','','The fixed-penalty experiment uses four designs with five saved development datasets each and two penalties; weights and offsets are controlled functions of observed predictors. The CV experiment uses actual fitted initial/target regressions and preliminary score offsets, the native five-fold/eight-level criterion, and both minimum/1SE policies. Original normalized weight units are preserved. Errors and warnings remain visible. The Julia solver is checked against the original objective and KKT conditions, not only coefficient differences. Native convergence tolerances allow small differences between numerical minimizers. CV selection differences or native failures must be investigated before substituting the solver.','','Source: calibration-loss.jl and corresponding reference/check runners. All numerical work ran on Discovery compute nodes; no core package/default changes.','']
(dest/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(checks))
