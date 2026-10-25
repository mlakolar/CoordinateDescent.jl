"""Aggregate exclusive Profile.print samples; retain the raw text profiles."""
import csv,os,re,statistics
from collections import defaultdict
from pathlib import Path
root=Path(os.environ['RUN'])/'results';rows=[];groups=defaultdict(dict)
for p in root.glob('profile-*.txt'):
 _,rep,case=p.stem.split('-');rep=int(rep);case=int(case)
 leaves=[]
 for line in p.read_text().splitlines():
  m=re.match(r'^\s*(\d+)\s+(\d+)\s+(\S+)\s+([?\d]+)\s+(.*)$',line)
  if m and int(m[2])>0:
   key=(m[3],m[4],m[5]);groups[(case,key)][rep]=groups[(case,key)].get(rep,0)+int(m[2]);leaves.append((int(m[2]),key))
 for rank,(count,key) in enumerate(sorted(leaves,reverse=True)[:10],1):
  rows.append(dict(rep=rep,case=case,rank=rank,exclusive_samples=count,file=key[0],line=key[1],function=key[2]))
with (root/'profile-top-leaves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for (case,key),counts in groups.items():
 summary.append(dict(case=case,replications_present=len(counts),median_exclusive_samples=statistics.median(counts.get(rep,0) for rep in range(1,8)),file=key[0],line=key[1],function=key[2]))
summary.sort(key=lambda r:(r['case'],-r['median_exclusive_samples']))
with (root/'profile-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Summarized',len(list(root.glob('profile-*.txt'))),'replicated profiles')
