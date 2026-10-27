import csv, pathlib, statistics, random, math, json
root=pathlib.Path(__file__).resolve().parent/'results'
rows=[]
for p in sorted(root.glob('timings-*.csv')):
    with p.open() as f: rows.extend(csv.DictReader(f))
assert len(rows)==7*2*(8*3*3+3),len(rows)
assert all(r['converged']=='true' for r in rows)
index={}
for r in rows:
    key=(r['case'],r['kind'],r['variant'],int(r['rep']))
    index.setdefault(key,[]).append(r)
rng=random.Random(9926)
table=[]
for case,kind in sorted(set((r['case'],r['kind']) for r in rows)):
    times={v:[statistics.median(float(x['seconds']) for x in index[case,kind,v,rep]) for rep in range(1,8)] for v in ('baseline','candidate')}
    ratios=[b/a for a,b in zip(times['baseline'],times['candidate'])]
    boots=sorted(statistics.median(rng.choices(ratios,k=7)) for _ in range(10000))
    alloc={v:statistics.median(statistics.median(int(x['bytes']) for x in index[case,kind,v,rep]) for rep in range(1,8)) for v in times}
    table.append(dict(case=case,kind=kind,baseline_seconds=statistics.median(times['baseline']),candidate_seconds=statistics.median(times['candidate']),paired_median_ratio=statistics.median(ratios),ratio_min=min(ratios),ratio_max=max(ratios),ratio_ci95=[boots[249],boots[9749]],baseline_bytes=alloc['baseline'],candidate_bytes=alloc['candidate'],paired_ratios=ratios))
(root/'summary.json').write_text(json.dumps(dict(rows=len(rows),datasets=7,rounds=3,table=table),indent=2))
with (root/'summary.csv').open('w') as f:
    keys=[k for k in table[0] if k!='paired_ratios']
    w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in table)
print(json.dumps(table,indent=2))
