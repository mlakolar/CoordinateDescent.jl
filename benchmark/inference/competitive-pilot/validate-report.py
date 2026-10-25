import csv,json,math
from pathlib import Path
rows=list(csv.DictReader(open('pilot-records.csv')))
assert len(rows)==720
assert sum(r['status']=='ok' for r in rows)==680
assert sum(r['status']=='unsupported' for r in rows)==40
coef=list(csv.DictReader(open('pilot-coefficients.csv')));assert len(coef)==6800
cases={c['scenario']:c for c in json.loads(Path('cases.json').read_text())}
cert=[r for r in rows if r['engine']=='sslasso' and r['mode']=='matched']
assert len(cert)==40
for r in cert:
 assert float(r['qp_max_residual_all'])<=cases[r['scenario']]['lambda_node']+1e-5,r
 assert float(r['qp_max_abs_gap_all'])<1e-4,r
parity=list(csv.DictReader(open('matched-nodewise-check.csv')));assert len(parity)==400
assert max(float(r['estimate_abs_difference']) for r in parity)<1e-4
assert max(float(r['se_abs_difference']) for r in parity)<1e-6
meta=list(Path('profiles').glob('*.meta'));assert len(meta)==5
assert all('same_seed_coefficient_check=passed' in p.read_text() for p in meta)
receipt=dict(supported_fits=680,unsupported_combinations=40,coefficient_rows=6800,
             matched_nodewise_comparisons=400,sslasso_corrected_qp_checks=40,
             sslasso_worst_all_row_gap=max(float(r['qp_max_abs_gap_all']) for r in cert),
             same_seed_profile_checks=5,full1000_comparison_submitted=False)
Path('validation-final.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
