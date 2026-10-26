"""Discovery compute-node explicit DoubleML denominators and Monte Carlo boundaries."""
import csv,json,math,statistics,collections
from pathlib import Path
D=Path('reports/development-doubleml-sparse');records=list(csv.DictReader((D/'records.csv').open()));cases=[r['case'] for r in csv.DictReader(Path('../doubleml-v2/doubleml-cases.csv').open())];methods=['doubleml_sparse_nuisance_space','doubleml_sparse_instrument']
def wilson(p,n):
 z=1.959963984540054;den=1+z*z/n;mid=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,mid-h),min(1,mid+h)]
grid=[]
for case in cases:
 for method in methods:
  ok=0
  for rep in range(1,21):
   rr=[r for r in records if r['case']==case and r['method']==method and int(r['rep'])==rep]
   ok+=len(rr)==2 and len({r['target'] for r in rr})==2 and all(r['converged'].lower()=='true' and float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) for r in rr)
  grid.append(dict(case=case,method=method,valid=ok,planned=20,availability=ok/20,availability_wilson95=wilson(ok/20,20)))
(D/'all-planned-outcomes.json').write_text(json.dumps(grid,indent=2)+'\n')
slots=json.loads((D/'per-slot.json').read_text());bounds=[]
for r in slots:
 n=r['valid_datasets']
 if n:bounds.append(dict(**r,coverage_wilson95=wilson(r['coverage'],n),rejection_wilson95=wilson(r['rejection'],n),note='Per-role independent-dataset Bernoulli Monte Carlo interval, conditional on complete valid score-datasets. No multiplicity adjustment.'))
(D/'role-monte-carlo-bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
resources=[]
for p in sorted(Path('resources').glob('doubleml-sparse-development-*.txt')):
 d=dict(l.split('=',1) for l in p.read_text().splitlines() if '=' in l)
 if all(k in d for k in ['cpu_seconds','max_rss_kib','process_seconds','process_exit']):resources.append({k:float(v) for k,v in d.items()})
res=dict(worker_receipts=len(resources),expected_workers=480,cpu_seconds=sum(r['cpu_seconds'] for r in resources),process_seconds_sum=sum(r['process_seconds'] for r in resources),max_worker_rss_gib=max(r['max_rss_kib'] for r in resources)/1024**2,note='Single dataset/score per fresh process, two requested targets. Actual child CPU;per-worker peakRSS,not aggregate memory. Allocation128CPUs/128GiB for3025s. Full results include121targeterrors;processsuccessisnot calibration.')
(D/'resource-summary.json').write_text(json.dumps(res,indent=2)+'\n')
counts=collections.Counter((e.get('error') or e.get('message') or str(e)) for e in json.loads((D/'errors.json').read_text()));(D/'error-categories.json').write_text(json.dumps(counts,indent=2)+'\n')
checks=json.loads((D/'checks.json').read_text());summary=json.loads((D/'strict-summary.json').read_text())
lines=['# Official DoubleMLLPLR sparse-learner pilot:20datasets per design','','EMPIRICAL. Native6884b5f1 framework with the declared sparse eight-penalty learners, nested five-by-five cross-fitting, both native scores, and two requested roles/dataset. This is not an author-bootstrap replication or a universal package-default comparison: DoubleML requires explicit nuisance learners.','','There are240datasets,480dataset/score workers and960planned target fits. The run returned839unique finite target records and121target errors, with62,925retained warning entries. All worker completion markers are present. Native score/sandwich SE parity is2.22e-16. All24planned case/method groups, including zero-success groups if any, are enumerated in all-planned-outcomes.json. Strict summaries require both requested target roles in a dataset.','','The moderate design has active coverage45%(nuisance_space) and55%(instrument), each on20/20complete datasets; MCSE is about11.4percentage points. The correlated instrument score gives95%active coverage and0%null rejection on20/20, with active length1.693 and power25%; its active-coverage MCSE is5percentage points. This is a favorable pilot result, not broad independent calibration. The larger-n design has only12/20complete datasets per score and active coverage58.3%/83.3%. The paper-style Liu n1000,p201 control gives100%active coverage in20/20for both scores, with90%null coverage. A20/20coverage result still has a Wilson95%Monte Carlo lower bound of83.9%, so zero empirical MCSE does not mean certainty.','','See STRICT_RESULTS.md for all designs, roles, power and interval length, and role-monte-carlo-bounds.json for per-role boundary-aware uncertainty. MC intervals here are conditional on successful complete datasets; availability is reported separately. A convergence warning may originate from a CV path level; the declared strict warning policy is conservative, not a failure theorem for DoubleML or all learners. Failed scientific fits were not silently retried.','','```json',json.dumps(res,indent=2),'```','','No logistic default or full logistic campaign is eligible from this pilot. The original broad-grid configuration and its failures remain separate. These results support testing favorable published regimes and improving nuisance estimation/finite-sample bias control; they do not justify truth-based variance inflation.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(dict(checks=checks,resources=res,planned_groups=len(grid))))
