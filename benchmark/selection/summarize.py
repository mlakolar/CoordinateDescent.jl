"""Summarize saved Discovery measurements without running any estimators."""
import csv, math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [row for f in sorted(root.glob('timing-*.csv')) for row in csv.DictReader(f.open())]
mc = [row for f in sorted(root.glob('mc-*.csv')) for row in csv.DictReader(f.open())]
assert len(rows) == 7*9, f'expected 63 timings, found {len(rows)}'
assert len(mc) == 7*50*4, f'expected 1400 Monte Carlo rows, found {len(mc)}'
assert len(list(root.glob('profile-*.txt'))) == 63
assert all(float(x['seconds'])>0 and math.isfinite(float(x['seconds'])) for x in rows)
assert len({(r['method'],r['seed']) for r in mc}) == len(mc)
lines = ['# Discovery validation of issue 82', '',
         'Seven independent process replications, two Julia threads and one BLAS thread per process. '
         'Wall times exclude warm-up and profiling. All nine workloads were profiled in every replication, '
         'for at least 0.2 seconds per workload. Tables summarize saved output; no estimators are rerun.', '',
         '| Workload | Median ms | Min–max ms | Median allocated MiB |',
         '|---|---:|---:|---:|']
for name in sorted({r['case'] for r in rows}):
    rs=[r for r in rows if r['case']==name]
    assert {int(r['rep']) for r in rs}==set(range(1,8))
    ts=[1000*float(r['seconds']) for r in rs]
    mem=[int(r['bytes'])/2**20 for r in rs]
    lines.append(f'| {name} | {statistics.median(ts):.3f} | {min(ts):.3f}–{max(ts):.3f} | {statistics.median(mem):.2f} |')
lines += ['', 'Monte Carlo uses 350 independent data/noise repetitions per procedure. '
          'The orthogonal known-noise model follows the SLOPE paper; normalized random fixed designs '
          'and a Gaussian AR(1) model with correlation 0.4 follow the knockoff design families. '
          'Dimensions are n=240, p=100, with 20 signals of size 4.5/sqrt(n) and sigma=1. '
          'These are bounded development experiments, not full reproductions of the papers.', '',
          '| Procedure | Mean FDP (MC SE) | Mean power (MC SE) | Mean false selections |',
          '|---|---:|---:|---:|']
for method in sorted({r['method'] for r in mc}):
    rs=[r for r in mc if r['method']==method]
    assert len(rs)==350
    vals={k:[float(r[k]) for r in rs] for k in ['fdp','power','false']}
    for r in rs:
        R,V=int(r['selected']),int(r['false'])
        assert 0<=V<=R<=100 and abs(float(r['fdp'])-V/max(1,R))<1e-12
        assert 0<=float(r['power'])<=1
    se=lambda k:statistics.stdev(vals[k])/math.sqrt(len(rs))
    lines.append(f'| {method} | {statistics.mean(vals["fdp"]):.4f} ({se("fdp"):.4f}) | '
                 f'{statistics.mean(vals["power"]):.4f} ({se("power"):.4f}) | {statistics.mean(vals["false"]):.3f} |')
lines += ['', 'SLOPE and both knockoff+ procedures target q=0.1 in these designs. '
          'CPSS uses 20 complementary pairs, a base cap q=10 and cutoff=0.8; its reported '
          'distribution-free low-selection-probability bound is 10²/(100×0.6)=1.667. '
          'Its FDP is descriptive. Interpreting its bound as PFER requires that all null '
          'variables have half-sample selection probability at most q/p. Monte Carlo is not a proof of FDR control.', '']
(root/'summary.md').write_text('\n'.join(lines))
print('\n'.join(lines))
