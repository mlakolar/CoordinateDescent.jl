import csv,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
out=root/'outputs'
final=root/'outputs-v2'
if (root/'results').is_dir():
    out=final=root/'results'
lines=['# Issue 77 Discovery validation','', 'Run: `20261025-cd77-inference`. All Julia execution and numerical/reference computations ran inside Slurm allocations.','']
if final.exists():
    groups={}
    for p in sorted(final.glob('p*-gram*.csv')):
        rows=list(csv.DictReader(p.open()))
        if len(rows)!=1: raise RuntimeError(str(p))
        r=rows[0]; assert r['converged']=='true'
        groups.setdefault((int(r['p']),r['gram']),[]).append(r)
    lines += ['Final timings use three independent design seeds, one untimed full-size warm-up, one timed call, and one separately profiled call per replication. Four Julia threads, one BLAS thread, n=200, AR(1) correlation 0.3.','', '| p | Gram | replications | median seconds | range seconds | median allocated MiB | median inverse nonzeros |','|---:|:---:|---:|---:|---:|---:|---:|']
    for (p,g),rs in sorted(groups.items()):
        t=[float(r['seconds']) for r in rs]; mem=[int(r['bytes'])/2**20 for r in rs]; nz=[int(r['nonzeros']) for r in rs]
        lines.append(f'| {p} | {g} | {len(rs)} | {statistics.median(t):.4f} | {min(t):.4f}–{max(t):.4f} | {statistics.median(mem):.1f} | {statistics.median(nz):.0f} |')
    lines += ['', 'Allocated bytes are cumulative allocations per call, not peak resident memory. Slurm accounting records peak memory separately. Initial v1 timing files include predictor-count-dependent sparse concatenation compilation and are retained as diagnostic results; they are excluded from final timing claims.','']
files=sorted(out.glob('coverage-*.csv')) if out.exists() else []
if files:
    lines += ['## Coverage','', 'Fixed n=100,p=500 Gaussian designs: Toeplitz correlation 0.9 and equicorrelation 0.8; support sizes 3/15, first/random support, fixed U[0,2]/U[0,4] signals. Each scenario has 100 independent error draws. Initial fit: universal scaled lasso; nodewise penalty 0.1 on population-standardized columns. These are the paper’s designs, with an explicit fixed nodewise penalty rather than its pooled-CV tuning. Simultaneous coverage and null FWER use 20 error replications per scenario and B=499 multipliers.','', '| design | s | support | c | reps | active coverage | null coverage | active length | simultaneous coverage | null FWER |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    total=0
    for p in files:
        rs=list(csv.DictReader(p.open())); assert len(rs)==100, str(p)
        assert all(r['converged']=='true' for r in rs)
        total+=len(rs); r=rs[0]
        number=lambda s: 1.0 if s=='true' else (0.0 if s=='false' else float(s))
        mean=lambda name:statistics.mean(number(x[name]) for x in rs if x[name]!='NaN')
        lines.append(f"| {r['design']} | {r['s']} | {r['placement']} | {r['c']} | {len(rs)} | {mean('active_coverage'):.3f} | {mean('null_coverage'):.3f} | {mean('active_length'):.3f} | {mean('simultaneous_coverage'):.3f} | {mean('fwer'):.3f} |")
    lines += ['',f'Validated coverage rows: {total}. Coverage fractions within a response draw are correlated across coefficients; the 100 response draws, rather than 100*p independent trials, are the replication unit. Simultaneous estimates have coarse Monte Carlo resolution (1/20). Poor coverage is an adverse finite-sample result, not a numerical convergence failure.','']
(root/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
