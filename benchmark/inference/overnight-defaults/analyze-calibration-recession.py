"""Compute-node aggregation of saved recession LP diagnostics; no refits."""
import json,csv
from pathlib import Path
rows=[];witness=[]
for p in sorted(Path('outputs/calibration-recession').glob('*/*/checks.json')):rows+=json.loads(p.read_text())
for p in sorted(Path('outputs/calibration-recession').glob('*/*/witnesses.json')):witness+=json.loads(p.read_text())
D=Path('reports/calibration-recession');D.mkdir(parents=True,exist_ok=True)
summary=[]
for case in sorted({r['case'] for r in rows}):
 for scope in ['actual_cv','fixed_reference']:
  rr=[r for r in rows if r['case']==case and r['scope']==scope]
  neg=[r for r in rr if r.get('unbounded_witness')]
  summary.append(dict(case=case,scope=scope,LPs=len(rr),LP_success=sum(r['lp_success'] for r in rr),negative_witnesses=len(neg),datasets_with_witness=len({r['rep'] for r in neg}),most_negative_recession_slope=min((r['recession_slope'] for r in neg),default=None),minimum_positive_response_direction=min((r['minimum_positive_response_direction'] for r in neg),default=None)))
fixed=[]
for r in rows:
 if r['scope']!='fixed_reference':continue
 tag='2' if r['level']==1 else '0.7';p=Path('outputs/calibration-engine')/r['case']/str(r['rep'])/f'rcal-{tag}.error'
 if p.exists():fixed.append(dict(**r,native_error=p.read_text()))
checks=dict(datasets=len({(r['case'],r['rep']) for r in rows}),linear_programs=len(rows),successful_LPs=sum(r['lp_success'] for r in rows),numerical_recession_witnesses=sum(r.get('unbounded_witness',False) for r in rows),native_uncertified_fixed_references=len(fixed),uncertified_fixed_references_with_witness=sum(r.get('unbounded_witness',False) for r in fixed))
for name,obj in [('checks',checks),('summary',summary),('fixed-reference-failures',fixed),('records',rows)]:(D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
lines=['# Calibrated-loss recession diagnostics','','Analytical criterion. For binary responses and nonnegative weights, take a direction u = intercept + X v. If u is nonnegative at every positive-response observation and mean(W (1-y) u) + lambda ||v||1 is negative, the penalized calibrated loss tends to minus infinity along that direction. The exponential positive-response terms remain bounded, while the negative-response linear term and penalty have strictly negative net slope. Finite offsets do not change this recession criterion.','','EMPIRICAL. Linear programs search for directions with bounded L1 norm. Each reported witness is recomputed in the original predictor coordinates. A small upward intercept adjustment gives positive-response direction slack; the resulting net slope must remain below -1e-8. These are numerical witnesses with saved coefficients, not a formal interval-arithmetic proof. A failed witness search is not proof of a finite minimizer.','','```json',json.dumps(checks,indent=2),'```','','|Case|Scope|LPs|Numerical negative witnesses|Datasets with witnesses|Most negative slope|','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['case']}|{r['scope']}|{r['LPs']}|{r['negative_witnesses']}|{r['datasets_with_witness']}|{r['most_negative_recession_slope']}|")
lines+=['','Actual-CV scope includes the full sample and all five training folds across eight penalties. Fixed-reference scope checks the two penalties in the independent RCAL/Julia engine experiment. Offsets/weights/targets are retained with those studies. A solver should reject an ill-posed level; increasing its iteration count cannot create a finite optimum. This diagnostic does not by itself identify why every slow solve is slow and does not establish inferential coverage. No package default was changed.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(checks))
assert checks['datasets']==20
