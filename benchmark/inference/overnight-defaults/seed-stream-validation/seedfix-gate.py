import json
from pathlib import Path
p=Path('reports/seedfix-validation-gaussian');checks=json.loads((p/'checks.json').read_text());gates=json.loads((p/'calibration-gates.json').read_text());roles=json.loads((p/'per-slot.json').read_text())
assert checks['planned_datasets']==5000 and checks['records']==50000 and checks['unique_records']==50000 and checks['fit_errors']==0 and not checks['missing_markers'] and checks['finite_records']
assert checks['max_noise_ratio']<=1.5*(1+1e-8)
assert len(gates)==10 and all(g['engineering_calibration_gate'] for g in gates)
flags=[r for r in roles if float(r['truth'])!=0 and r['coverage']-1.959963984540054*r['coverage_mcse']<.925]
(p/'role-flags.json').write_text(json.dumps(flags,indent=2)+'\n');print('Pooled gates passed; role review still required:',len(flags),'flags')
