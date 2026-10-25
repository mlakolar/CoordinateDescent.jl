import csv,json,math,runpy,sys
from pathlib import Path
stage,maxrep=sys.argv[1],int(sys.argv[2]);casefile='../calibration-v2/calibration-cases.csv';cases=list(csv.DictReader(open(casefile)))
records=[];diagnostics=[];errors=[];warnings=[];missing=[]
for case in cases:
    root=Path('outputs')/stage/case['case']
    for rep in range(1,maxrep+1):
        for slot in range(1,int(case['target_count'])+1):
            stem=root/(str(rep)+'.slot'+str(slot))
            if not Path(str(stem)+'.cal.complete').exists():missing.append([case['case'],rep,slot])
            for suffix,container in [('.cal.csv',records),('.cal.diagnostics.csv',diagnostics)]:
                f=Path(str(stem)+suffix)
                if f.exists():
                    for r in csv.DictReader(f.open()):r.update(case=case['case'],family='binomial',rep=rep);container.append(r)
    for f in root.glob('*.cal.error'):errors.append(dict(case=case['case'],file=f.name,error=f.read_text()))
    for f in root.glob('*.cal.warnings'):warnings.append(dict(case=case['case'],file=f.name,warning=f.read_text()))
dest=Path('reports')/(stage+'-calibration');dest.mkdir(parents=True,exist_ok=True)
for name,rows in [('records',records),('diagnostics',diagnostics)]:
    if rows:
        with (dest/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
keys=[(r['case'],r['rep'],r['method'],r['target']) for r in records]
checks=dict(stage=stage,planned_datasets=len(cases)*maxrep,records=len(records),unique_records=len(set(keys)),missing_markers=missing,target_errors=len(errors),warning_files=len(warnings),finite_records=all(float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) for r in records),max_alpha_kkt=max([float(r['alpha_kkt']) for r in diagnostics],default=None),max_gamma_kkt=max([float(r['gamma_kkt']) for r in diagnostics],default=None),max_root_error=max([float(r['root_residual']) for r in diagnostics],default=None),max_derivative_error=max([float(r['derivative_error']) for r in diagnostics],default=None))
for name,obj in [('checks',checks),('errors',errors),('warnings',warnings)]: (dest/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(checks))
if missing or not records or len(keys)!=len(set(keys)) or not checks['finite_records']:raise SystemExit(1)
sys.argv=['summarize-slots.py',stage,'calibration',str(maxrep),casefile]
runpy.run_path('../calibration-v2/summarize-slots.py',run_name='__main__')
if stage=='canary':
    counts=json.loads((dest/'strict-counts.json').read_text())
    for method in ['calibrated_min','calibrated_1se']:
        if not any(r['case']=='liu-published' and r['method']==method and r['valid']==maxrep for r in counts):raise SystemExit('published-design positive-control method unavailable: '+method)
