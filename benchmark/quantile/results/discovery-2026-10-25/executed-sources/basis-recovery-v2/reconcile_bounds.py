"""Use shared feasible dual lower bounds for the identical saved problem."""
import csv,json,math,shutil,subprocess,sys,itertools
from pathlib import Path
root=Path(sys.argv[1]).resolve();dest=root/'shared-bounds';dest.mkdir(exist_ok=True)
studies={'main-results':'summary','pilot-v2':'pilot-summary','admm-results':'admm-summary','rbench-results':'rbench-summary','rho-results':'rho-summary','split-results':'split-summary','basis-results':'basis-summary'}
def key(r):return r['case'],r['seed'],float(r['lambda'])
def bounds_for(rows):
    best={}
    for r in rows:
        d=float(r['dual'])
        if math.isfinite(d) and (key(r) not in best or d>best[key(r)][0]):best[key(r)]=(d,r['method'],r['rep'])
    return best
controls=[dict(case='a',seed=seed,**{'lambda':lam,'dual':d,'method':m,'rep':'1'}) for seed,lam,d,m in [('1','.1','1','a'),('1','.1','2','b'),('1','.2','20','c'),('2','.1','50','d')]]
cb=bounds_for(controls)
assert cb[('a','1',.1)]==(2.,'b','1') and cb[('a','1',.2)][0]==20. and cb[('a','2',.1)][0]==50.
audits={}
for data,summary in studies.items():
    src=root/('basis-completed' if data=='basis-results' and (root/'basis-completed').exists() else data)
    files=sorted(src.glob('*-solutions.csv'))
    if not files:continue
    blocks=[]
    for f in files:
        rows=list(csv.DictReader(f.open()));blocks.append((f,rows))
    best=bounds_for(itertools.chain.from_iterable(rows for _,rows in blocks))
    dst=dest/data;dst.mkdir(exist_ok=True);changes=0;largest=0.;negative=[]
    for f,rows in blocks:
        for r in rows:
            oldgap=float(r['gap'])
            r['original_dual']=r['dual'];r['original_gap']=r['gap'];r['best_bound_method']='';r['best_bound_rep']=''
            if key(r) not in best or not math.isfinite(float(r['primal'])):continue
            d,method,rep=best[key(r)]
            r['best_bound_method']=method;r['best_bound_rep']=rep
            r['dual']=str(d);r['gap']=str(float(r['primal'])-d)
            newgap=float(r['gap'])
            if math.isfinite(oldgap):assert newgap<=oldgap+1e-10
            if newgap<oldgap-1e-10:changes+=1;largest=max(largest,oldgap-newgap)
            if newgap < -1e-7:negative.append({k:r[k] for k in ['case','seed','method','k','gap']})
        if rows:
            with (dst/f.name).open('w') as out:
                w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for f in src.glob('*-paths.csv'):shutil.copy2(f,dst/f.name)
    for f in list(src.glob('*-choice.txt'))+list(src.glob('tuning.csv')):shutil.copy2(f,dst/f.name)
    if negative:raise RuntimeError(f'Invalid shared bounds for {data}: {negative[:3]}')
    subprocess.run([sys.executable,str(root/'summarize.py'),str(dst),str(dest/summary)],check=True)
    audits[data]={'solution_rows':sum(len(r) for _,r in blocks),'distinct_problems':len(best),'bounds_improved':changes,'max_gap_reduction':largest,'negative_gap_failures':len(negative)}
for name in ['summarize.py','approximation.csv','profiles','density-results']:
    target=dest/name
    if not target.exists() and (root/name).exists():target.symlink_to(root/name,target_is_directory=(root/name).is_dir())
(dest/'bound-audit.json').write_text(json.dumps(audits,indent=2))
subprocess.run([sys.executable,str(root/'report_inputs.py'),str(dest)],check=True)
print(json.dumps(audits,indent=2))
