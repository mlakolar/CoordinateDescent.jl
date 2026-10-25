"""Validate saved coverage and prepare report inputs without rerunning fits."""
import csv,json,sys,statistics as st
from pathlib import Path
root=Path(sys.argv[1]);out=root/'report-inputs';out.mkdir(exist_ok=True)
expected={'summary':(552,92,24),'pilot-summary':(46,23,1),'admm-summary':(72,12,12),'rbench-summary':(90,15,9)}
checks={};summaries={}
for folder,(paths,groups,files) in expected.items():
    f=root/folder
    if not (f/'coverage.json').exists():
        checks[folder]={'present':False};continue
    c=json.loads((f/'coverage.json').read_text());s=list(csv.DictReader((f/'summary.csv').open()))
    checks[folder]={'present':True,'counts':c,'expected_paths':paths,'expected_groups':groups,'expected_files':files,'coverage_complete':c['paths']==paths and c['groups']==groups and c['files']==files}
    summaries[folder]=s
    (out/(folder+'.md')).write_text((f/'TABLE.md').read_text())
# Density uses the primal LP formulation throughout; this is a study of the
# QR-style formulation's grid scaling, not an entire-breakpoint algorithm.
f=root/'density-results/density.csv'
if f.exists():
    rows=list(csv.DictReader(f.open()));groupset={}
    for row in rows:groupset.setdefault((row['method'],row['levels']),[]).append(row)
    ds=[]
    for (method,levels),rs in sorted(groupset.items()):
        valid=[r for r in rs if r['status']=='complete' and -1e-7<=float(r['maxgap'])<=1e-5]
        pm={}
        for r in valid:pm.setdefault(r['seed'],[]).append(float(r['total_s']))
        ms=[st.mean(v) for v in pm.values()]
        ds.append({'method':method,'levels':int(levels),'valid':len(valid),'attempted':len(rs),'input_count':len(ms),'mean_s':st.mean(ms) if ms else None,'input_sd_s':st.stdev(ms) if len(ms)>1 else None,'mean_pivots':st.mean(float(r['pivots']) for r in valid) if valid else None})
    (out/'density-summary.json').write_text(json.dumps(ds,indent=2));checks['density']={'rows':len(rows),'expected_rows':108,'coverage_complete':len(rows)==108}
f=root/'approximation.csv'
if f.exists():
    rows=list(csv.DictReader(f.open()));qa=[]
    for case in sorted({r['case'] for r in rows}):
        rs=[r for r in rows if r['case']==case]
        qa.append({'case':case,'rows':len(rs),'max_check_excess':max(float(r['check_excess']) for r in rs),'min_check_excess':min(float(r['check_excess']) for r in rs),'max_training_prediction_rms':max(float(r['training_prediction_rms']) for r in rs),'max_coefficient_l2':max(float(r['coefficient_l2']) for r in rs),'min_smooth_objective_gain':min(float(r['smooth_objective_gain']) for r in rs)})
    (out/'approximation-summary.json').write_text(json.dumps(qa,indent=2));checks['approximation']={'rows':len(rows),'expected_rows':144,'coverage_complete':len(rows)==144}
(out/'coverage.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
