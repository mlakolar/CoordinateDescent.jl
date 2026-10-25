"""Validate saved coverage and prepare report inputs without rerunning fits."""
import csv,json,sys,statistics as st,subprocess,re
from pathlib import Path
root=Path(sys.argv[1]);out=root/'report-inputs';out.mkdir(exist_ok=True)
expected={'summary':(552,92,24),'pilot-summary':(46,23,1),'admm-summary':(72,12,12),'rbench-summary':(90,15,9),'rho-summary':(24,4,6),'split-summary':(54,9,9),'basis-summary':(96,16,12)}
checks={};summaries={}
for folder,(paths,groups,files) in expected.items():
    f=root/folder
    if folder=='rho-summary':
        eligible_cases=sorted(p.name.removesuffix('-choice.txt') for p in (root/'rho-results').glob('*-choice.txt'))
        paths,groups,files=12*len(eligible_cases),2*len(eligible_cases),3*len(eligible_cases)
        checks['rho-eligibility']={'eligible_cases':eligible_cases,'reason':'Confirmation is conditional on an eligible full-path tuning choice; failed pilot configurations remain in tuning.csv.'}
    datafolder={'summary':'main-results','pilot-summary':'pilot-v2','admm-summary':'admm-results','rbench-summary':'rbench-results','rho-summary':'rho-results','split-summary':'split-results','basis-summary':'basis-results'}[folder]
    if not (f/'coverage.json').exists() and list((root/datafolder).glob('*-paths.csv')):
        subprocess.run([sys.executable,str(root/'summarize.py'),str(root/datafolder),str(f)],check=True)
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
        qa.append({'case':case,'rows':len(rs),'max_check_excess':max(float(r['check_excess']) for r in rs),'min_check_excess':min(float(r['check_excess']) for r in rs),'max_training_prediction_rms':max(float(r['training_prediction_rms']) for r in rs),'max_coefficient_l2':max(float(r['coefficient_l2']) for r in rs),'max_smoothed_dual_gap':max(float(r['smoothed_dual_gap']) for r in rs),'min_smooth_objective_gain':min(float(r['smooth_objective_gain']) for r in rs)})
    (out/'approximation-summary.json').write_text(json.dumps(qa,indent=2));checks['approximation']={'rows':len(rows),'expected_rows':144,'coverage_complete':len(rows)==144}
(out/'coverage.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))

# Paired input contrasts; retain only comparisons whose entire timing groups
# pass the common post-fit gate. Within-process repetitions are averaged first.
contrasts=[]
for folder,datafolder in [('summary','main-results'),('admm-summary','admm-results'),('rbench-summary','rbench-results'),('rho-summary','rho-results'),('split-summary','split-results'),('basis-summary','basis-results')]:
    if folder not in summaries:continue
    eligible={(s['case'],s['method']) for s in summaries[folder] if int(s['valid_paths'])==int(s['total_paths'])==6 and int(s['independent_inputs'])==3}
    raw=[]
    for p in (root/datafolder).glob('*-paths.csv'):raw.extend(csv.DictReader(p.open()))
    times={}
    for r in raw:
        if (r['case'],r['method']) in eligible:times.setdefault((r['case'],r['method'],r['seed']),[]).append(float(r['total_s']))
    pairs=[('lp-dual-dual-cold','lp-dual-dual-warm'),('gurobi-primal-dual-warm','gurobi-primal-primal-warm'),('gurobi-primal-dual-warm','lp-dual-dual-warm'),('apg-0.2','newton-0.2'),('linear-admm','block-admm'),('lp-dual-dual-warm','block-admm'),('lp-dual-dual-warm','quantreg-fn'),('gurobi-primal-primal-warm','quantreg-fn'),('conquer','newton-0.2'),('lp-dual-dual-warm','tuned-admm'),('lp-dual-dual-warm','lp-split-primal-warm'),('lp-dual-dual-warm','lp-split-dual-warm'),('lp-dual-dual-warm','lp-dual-nullbasis-warm')]
    for case in sorted({s['case'] for s in summaries[folder]}):
        for baseline,candidate in pairs:
            if (case,baseline) not in eligible or (case,candidate) not in eligible:continue
            seeds=sorted({k[2] for k in times if k[:2]==(case,baseline)} & {k[2] for k in times if k[:2]==(case,candidate)})
            if len(seeds)!=3:continue
            bm=[st.mean(times[(case,baseline,s)]) for s in seeds];cm=[st.mean(times[(case,candidate,s)]) for s in seeds]
            ds=[b-c for b,c in zip(bm,cm)];rat=[b/c for b,c in zip(bm,cm)]
            se=st.stdev(ds)/(len(ds)**.5);rse=st.stdev(rat)/(len(rat)**.5)
            contrasts.append(dict(study=folder,case=case,baseline=baseline,candidate=candidate,input_count=3,mean_saved_seconds=st.mean(ds),saved_t95_lo=st.mean(ds)-4.303*se,saved_t95_hi=st.mean(ds)+4.303*se,mean_input_time_ratio=st.mean(rat),ratio_t95_lo=st.mean(rat)-4.303*rse,ratio_t95_hi=st.mean(rat)+4.303*rse))
(out/'paired-contrasts.json').write_text(json.dumps(contrasts,indent=2))

profiles=[]
for p in sorted((root/'profiles').glob('*.txt')):
    totals={};snapshots=None;self_total=0
    lines=p.read_text().splitlines()
    for line in lines:
        m=re.match(r"\s*(\d+)\s+(\d+)\s+(.+)",line)
        if m:
            count=int(m.group(2));text=m.group(3)
            if 'libhighs' in text:cat='native HiGHS'
            elif 'gurobi' in text.lower() and '.so' in text:cat='native Gurobi'
            elif 'openblas' in text:cat='BLAS'
            elif 'gc.c' in text or 'gc-' in text or 'gc_' in text:cat='GC'
            elif 'ProximalSolvers' in text or 'CoordinateDescent' in text:cat='production solver Julia'
            elif 'ProximalBase' in text:cat='loss/penalty Julia'
            else:cat='other Julia/runtime'
            totals[cat]=totals.get(cat,0)+count;self_total+=count
        m=re.search(r'Total snapshots: (\d+)',line)
        if m:snapshots=int(m.group(1))
    profiles.append({'file':p.name,'metadata':lines[0] if lines else '', 'snapshots':snapshots,'sum_self_counts':self_total,'self_counts':totals})
(out/'profile-summary.json').write_text(json.dumps(profiles,indent=2))
