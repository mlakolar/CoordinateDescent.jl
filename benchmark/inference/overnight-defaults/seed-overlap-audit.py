import csv,json,itertools
from pathlib import Path
old=list(csv.DictReader(Path('../source-v2/cases.csv').open()))
confirm=list(csv.DictReader(Path('../gaussian-hc3-v1/confirmation-cases.csv').open()))
stages=[('canary',92000000,1,old),('development',92100000,100,old),('confirmation',92200000,500,[c for c in confirm if c['family']=='gaussian']),('campaign',92300000,1000,[c for c in confirm if c['family']=='gaussian']),('new_validation',1000000000,500,[c for c in confirm if c['family']=='gaussian'])]
# Preserve original physical rows rather than renumbering Gaussian subsets.
index={c['case']:i for i,c in enumerate(confirm,1)}
ranges=[]
for stage,base,count,cases in stages:
 for c in cases:
  row=index[c['case']];ranges.append(dict(stage=stage,case=c['case'],row=row,start=base+row*10000+1,end=base+row*10000+count))
pairs=[]
for a,b in itertools.combinations(ranges,2):
 if a['stage']==b['stage']:continue
 lo=max(a['start'],b['start']);hi=min(a['end'],b['end'])
 if lo<=hi:pairs.append(dict(a=a,b=b,overlap_seed_count=hi-lo+1,first=lo,last=hi))
assert not any(p['a']['stage']=='new_validation' or p['b']['stage']=='new_validation' for p in pairs)
out=Path('reports/seed-stream-audit');out.mkdir(exist_ok=True,parents=True)
(out/'ranges.json').write_text(json.dumps(ranges,indent=2)+'\n');(out/'overlaps.json').write_text(json.dumps(pairs,indent=2)+'\n')
(out/'RESULTS.md').write_text('# Seed-stream provenance correction\n\nMETADATA AUDIT. The earlier stage bases differ by100000 while expanded physical design rows span200000. The saved overlaps.json explicitly enumerates cross-stage equal seed intervals. Equal seeds across different designs need not give identical responses or matrices, but disjoint random streams were not ensured. Per-design estimates and failure records remain intact. Strong claims that the complete earlier confirmation collection was independent of all development streams are qualified.\n\nNew validation base1000000000 is disjoint from these audited blocks. Its source was frozen and pushed3eefe3b before any new dataset was generated. The Gaussian statistical tuning rule was not changed. Earlier historical base95000000 confirmation canary and base92500000 campaign canary are also separated from this new block. Reusing deterministic fitting RNG970000+rep across stages does not duplicate response datasets. This is a seed-metadata audit, not a test of probabilistic independence or proof about PRNGs.\n')
print('Cross-stage overlapping ranges:',len(pairs),'new-validation overlaps:0')
