"""Summarize already-collected replicated profiles; never rerun a fit."""
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

out=Path(os.environ['RUN'])/'results'
groups=defaultdict(list)
leaves=[]
for p in sorted(out.glob('profile-*.txt')):
    m=re.match(r'profile-(\d+)-(.+?)-(CD-threaded|CD|SLOPE\.jl|Knockoffs\.jl|GLMNet\.jl)\.txt$',p.name)
    if not m: raise ValueError(p)
    rep,case,engine=m.groups()
    active=[]
    for line in p.read_text().splitlines():
        row=re.match(r'^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\d+|\?)\s+(.*)$',line)
        if not row: continue
        count,overhead,file,lineno,function=row.groups()
        overhead=int(overhead)
        if not overhead: continue
        if any(s in file for s in ('task.jl','condition.jl','asyncevent.jl','partr.jl')): continue
        if any(s in function for s in ('poptask','task_done','pthread_cond','wait(')): continue
        active.append((overhead,file,lineno,function))
    top=sorted(active,reverse=True)[:5]
    groups[case,engine].append((int(rep),top))
    for overhead,file,line,function in top:
        leaves.append(dict(rep=rep,case=case,engine=engine,leaf_samples=overhead,file=file,line=line,function=function))
summary=[]
for (case,engine),rs in sorted(groups.items()):
    assert len(rs)==7,(case,engine,len(rs))
    counts=Counter((file,line,function) for rep,top in rs for overhead,file,line,function in top)
    for (file,line,function),n in counts.most_common(5):
        summary.append(dict(case=case,engine=engine,replications_in_top_five=n,file=file,line=line,function=function))
for name,data in [('profile-leaves.csv',leaves),('profile-summary.csv',summary)]:
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(f'PROFILE_SUMMARY_DONE {sum(len(v) for v in groups.values())} profiles, {len(groups)} case/engine groups')
