import csv,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
inputs=root/'outputs-glm-v2' if (root/'outputs-glm-v2').is_dir() else (root/'outputs-glm' if (root/'outputs-glm').is_dir() else root/'results')
rows=[]; lines=['# High-dimensional logistic validation','', 'Discovery: 20 response replications in each of 12 fixed Toeplitz n=100,p=500 designs (rho=0.9). Signal sizes 3/15, first/random support, fixed uniform coefficients on [0,c], c=1/2/4. Initial logistic penalty 0.05; Hessian-weighted nodewise penalty 0.1 on standardized columns. Null decorrelated score: nuisance penalty 0.05, projection penalty 0.03. All settings and seeds were fixed before the run.','', '| s | support | c | valid fits / attempted | active coverage | null coverage | active length | valid scores | null rejection rate | errors |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for path in sorted(inputs.glob('glmcoverage-*.csv'),key=lambda p:int(p.stem.split('-')[1])):
    rs=list(csv.DictReader(path.open())); assert len(rs)==20,str(path)
    good=[r for r in rs if r['fit_converged']=='true']; scores=[r for r in rs if r['score_converged']=='true']
    mean=lambda name:statistics.mean(float(r[name]) for r in good) if good else float('nan')
    rate=statistics.mean(float(r['null_score_pvalue'])<=.05 for r in scores) if scores else float('nan')
    r=rs[0]; errors=sum(bool(r['error']) for r in rs)
    lines.append(f"| {r['s']} | {r['placement']} | {r['c']} | {len(good)} / {len(rs)} | {mean('active_coverage'):.3f} | {mean('null_coverage'):.3f} | {mean('active_length'):.3f} | {len(scores)} | {rate:.3f} | {errors} |")
    rows.extend(rs)
assert len(rows)==240,len(rows)
lines+=['','Coverage is conditional on numerically converged fits; any failed fits remain in the raw CSVs and denominators above. Score rejection uses only converged score fits, with its denominator displayed. Twenty replications give coarse Monte Carlo resolution and do not establish reliable nominal calibration. These fixed penalties differ from the paper’s CV tuning. In particular, strong signals and weak effective information can make the asymptotic approximation inaccurate.','']
(root/'glm-summary-v2.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
