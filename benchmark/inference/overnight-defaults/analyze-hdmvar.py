"""Compute-node aggregation; require complete target sets before dataset summaries."""
import csv,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
stage,maxrep=sys.argv[1],int(sys.argv[2]);cases=[r for r in csv.DictReader(open('cases.csv')) if r['family']=='binomial']
records=[];errors=[];missing=[];diagnostics=[];summary=[];per_target=[];invalid=[]
for case in cases:
    root=Path('outputs')/stage/case['case']
    for rep in range(1,maxrep+1):
        stem=root/str(rep)
        if not stem.with_suffix('.hdmvar.complete').exists():missing.append([case['case'],rep])
        file=stem.with_suffix('.hdmvar.csv')
        if file.exists():
            for r in csv.DictReader(file.open()):
                r.update(case=case['case'],family=case['family'],rep=rep);records.append(r)
        file=stem.with_suffix('.hdmvar.diagnostics.csv')
        if file.exists():
            for r in csv.DictReader(file.open()):r.update(case=case['case'],rep=rep);diagnostics.append(r)
    for f in root.glob('*.hdmvar.error'):errors.append(dict(case=case['case'],file=f.name,error=f.read_text()))
groups=defaultdict(lambda:defaultdict(list))
for r in records:groups[(r['case'],r['method'])][r['rep']].append(r)
def summarize(rr):
    return dict(coverage=statistics.mean(float(float(r['lower'])<=float(r['truth'])<=float(r['upper'])) for r in rr),rejection=statistics.mean(float(float(r['pvalue'])<.05) for r in rr),length=statistics.mean(float(r['upper'])-float(r['lower']) for r in rr),signed_z=statistics.mean((float(r['estimate'])-float(r['truth']))/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr))
for (case,method),reps in sorted(groups.items()):
    c=next(x for x in cases if x['case']==case)
    # Belloni tests d plus five nulls; SIHR two active plus five nulls; other cases five plus five.
    count=6 if c['design']=='belloni' else 7 if c['design']=='sihr' else 10
    valid={}
    for rep,rr in reps.items():
        ok=len(rr)==count and len({r['target'] for r in rr})==count and all(r['converged'].lower()=='true' and float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) and 0<=float(r['pvalue'])<=1 for r in rr)
        if ok:valid[rep]=rr
        else:invalid.append([case,method,rep,len(rr)])
    for cls in ['active','null']:
        vals=[summarize([r for r in rr if (float(r['truth'])!=0)==(cls=='active')]) for rr in valid.values() if any((float(r['truth'])!=0)==(cls=='active') for r in rr)]
        if not vals:continue
        out=dict(case=case,method=method,target_class=cls,valid_datasets=len(vals),planned_datasets=maxrep)
        for key in vals[0]:
            v=[x[key] for x in vals];out[key]=statistics.mean(v);out[key+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
        summary.append(out)
    targets=sorted({r['target'] for rr in valid.values() for r in rr},key=int)
    for target in targets:
        rr=[r for rows in valid.values() for r in rows if r['target']==target]
        out=dict(case=case,method=method,target=target,truth=rr[0]['truth'],valid_datasets=len(rr),planned_datasets=maxrep,**summarize(rr));per_target.append(out)
keys=[(r['case'],r['rep'],r['method'],r['target']) for r in records]
checks=dict(stage=stage,planned_datasets=len(cases)*maxrep,records=len(records),unique_records=len(set(keys)),missing_markers=missing,method_target_errors=len(errors),incomplete_or_invalid_datasets=invalid,finite_records=all(float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','pvalue','lower','upper']) for r in records),max_instrument_coefficient_difference=max([float(r['coefficient_parity']) for r in diagnostics if r['coefficient_parity']!='NA'],default=None),max_instrument_se_difference=max([float(r['se_parity']) for r in diagnostics if r['se_parity']!='NA'],default=None))
dest=Path('reports')/(stage+'-hdmvar');dest.mkdir(exist_ok=True,parents=True)
for name,value in [('checks',checks),('summary',summary),('per-target',per_target),('errors',errors)]: (dest/(name+'.json')).write_text(json.dumps(value,indent=2))
for name,rr in [('records',records),('diagnostics',diagnostics)]:
    if rr:
        with (dest/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
lines=['# Native hdm logistic variance audit','','EMPIRICAL. Unchanged native hdm selection/refit coefficients; full selected-model Hessian and HC1 sandwich comparisons. Native S22 accesses l3$index, which standard glm does not supply. Audit candidates are distinct from the native default. Complete-target, converged datasets only; valid/planned counts include target-specific failures. HC1 uses n/(n-k-1), where k is initial support size. Logistic base score SE is model based; HC1 score uses residual sandwich. All times are diagnostic pipeline costs with shared computations, not competitor benchmarks.','','|Case|Method|Class|Valid/planned|Coverage (MCSE)|Rejection (MCSE)|Length|','|---|---|---|---|---|---|---|']
for r in summary:lines.append(f"|{r['case']}|{r['method']}|{r['target_class']}|{r['valid_datasets']}/{r['planned_datasets']}|{r['coverage']:.4f} ({r['coverage_mcse']})|{r['rejection']:.4f} ({r['rejection_mcse']})|{r['length']:.4f}|")
(dest/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(checks))
if missing or len(keys)!=len(set(keys)) or not checks['finite_records']:raise SystemExit(1)
if stage=='canary' and not all(any(r['case']==c['case'] and r['method']=='hdm_glm_model' for r in summary) for c in cases):raise SystemExit('missing canary baseline')
