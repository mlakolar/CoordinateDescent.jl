import csv,json,math,runpy,sys
from pathlib import Path
stage,maxrep=sys.argv[1],int(sys.argv[2]);casefile='../gaussian-campaign-v1/confirmation-cases.csv';cases=[c for c in csv.DictReader(open(casefile)) if c['family']=='gaussian']
records=[];diagnostics=[];errors=[];missing=[]
for case in cases:
    root=Path('outputs')/stage/case['case']
    for rep in range(1,maxrep+1):
        stem=root/str(rep)
        if not stem.with_suffix('.gaussian.complete').exists():missing.append([case['case'],rep])
        for suffix,container in [('.gaussian.csv',records),('.gaussian.diagnostics.csv',diagnostics)]:
            f=stem.with_suffix(suffix)
            if f.exists():
                for r in csv.DictReader(f.open()):r.update(case=case['case'],family='gaussian',rep=rep);container.append(r)
    for f in root.glob('*.gaussian.error'):errors.append(dict(case=case['case'],file=f.name,error=f.read_text()))
dest=Path('reports')/(stage+'-gaussian');dest.mkdir(parents=True,exist_ok=True)
for name,rr in [('records',records),('diagnostics',diagnostics)]:
    if rr:
        with (dest/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
keys=[(r['case'],r['rep'],r['method'],r['target']) for r in records]
checks=dict(stage=stage,planned_datasets=len(cases)*maxrep,records=len(records),unique_records=len(set(keys)),missing_markers=missing,fit_errors=len(errors),finite_records=all(float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) for r in records),max_noise_ratio=max([float(r['noise_ratio']) for r in diagnostics],default=None),max_leverage_trace_error=max([float(r['leverage_trace_error']) for r in diagnostics],default=None))
for name,obj in [('checks',checks),('errors',errors)]: (dest/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(checks))
if missing or not records or len(keys)!=len(set(keys)) or not checks['finite_records']:raise SystemExit(1)
sys.argv=['summarize-slots.py',stage,'gaussian',str(maxrep),casefile]
runpy.run_path('../gaussian-campaign-v1/summarize-slots.py',run_name='__main__')
summary=json.loads((dest/'strict-summary.json').read_text());counts=json.loads((dest/'strict-counts.json').read_text())
gates=[]
if maxrep>=500:
    for case in cases:
        c=case['case'];count=next((r for r in counts if r['case']==c),None)
        active=next((r for r in summary if r['case']==c and r['target_class']=='active'),None);null=next((r for r in summary if r['case']==c and r['target_class']=='null'),None)
        lower=active['coverage']-1.959963984540054*active['coverage_mcse'] if active else None
        upper=null['rejection']+1.959963984540054*null['rejection_mcse'] if null else None
        passed=bool(count and count['valid']/maxrep>=.99 and lower is not None and lower>=.925 and upper<=.075)
        gates.append(dict(case=c,valid=count['valid'] if count else 0,planned=maxrep,active_lower95_mc=lower,null_upper95_mc=upper,engineering_calibration_gate=passed))
(dest/'calibration-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
# A passed numerical job never auto-promotes defaults or conceals a failed gate.
