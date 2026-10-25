#!/usr/bin/env python3
"""Aggregate already-measured CSVs; never refit or benchmark."""
import csv, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [r for p in sorted(root.glob('rep*.csv')) for r in csv.DictReader(p.open())]
assert len(rows) == 225, f'expected 5 seeds × 3 sizes × 5 methods × 3 rounds, got {len(rows)}'
assert all(r['converged']=='true' for r in rows), 'nonconverged measured fit'
keys={(int(r['rep']),int(r['q']),r['method'],int(r['round'])) for r in rows}
assert len(keys)==len(rows), 'duplicate timing identity'
profiles=list(root.glob('profile-rep*-q*-*.txt'))
assert len(profiles)==75, f'expected 75 profiles, got {len(profiles)}'
assert all(p.stat().st_size>0 for p in profiles), 'empty profile'
print('| q | method | median seconds | replication range | median allocated MB |')
print('|---:|---|---:|---:|---:|')
for q in (10,100,1000):
 for method in ('rank','nuclear','nuclear_path','sparse','sparse_svd'):
  cells=[r for r in rows if int(r['q'])==q and r['method']==method]
  perrep=[statistics.median(float(r['seconds']) for r in cells if int(r['rep'])==rep) for rep in range(1,6)]
  mb=statistics.median(int(r['bytes']) for r in cells)/1e6
  print(f'| {q} | {method} | {statistics.median(perrep):.5g} | {min(perrep):.5g}–{max(perrep):.5g} | {mb:.3g} |')
print('\nAll 225 measurements converged. Five independent data replications, three rounds each.')
print('Profiles: 75 nonempty text reports (five data replications for each method and response size).')
