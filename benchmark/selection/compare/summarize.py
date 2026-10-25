"""Aggregate retained output only. Execute on Discovery inside the campaign job."""
import csv
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

root = Path(os.environ['RUN'])
out = root / 'results'
expected_reps = int(os.environ.get('EXPECTED_REPS', '7'))
expected_b = int(os.environ.get('MC_B', '20'))
scenarios = ['iid', 'AR(.4)', 'AR(.8)', 'weak AR(.4)', 'null AR(.4)', 'wide AR(.4)']

def rows(pattern):
    return [r for p in sorted(out.glob(pattern)) if 'summary' not in p.name for r in csv.DictReader(p.open())]

def stats(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return (st.mean(xs), st.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float('nan')) if xs else (float('nan'), float('nan'))

timings = rows('timing-*.csv')
byproc = defaultdict(list)
for r in timings:
    t=float(r['seconds'])
    assert math.isfinite(t) and t>0, r
    byproc[r['case'],r['engine'],int(r['rep'])].append(t)
medians = {key:st.median(values) for key,values in byproc.items()}
assert all(len(v)==3 for v in byproc.values()), 'expected three timing batches per process/case'
groups = defaultdict(list)
for (case,engine,rep),value in medians.items():
    groups[case,engine].append(value)
assert len(groups)==30, len(groups)
timing_summary=[]
for (case,engine),xs in sorted(groups.items()):
    assert len(xs)==expected_reps,(case,engine,len(xs))
    timing_summary.append(dict(case=case,engine=engine,replications=len(xs),median_ms=1000*st.median(xs),min_ms=1000*min(xs),max_ms=1000*max(xs)))

comparisons=[]
for (case,engine),xs in sorted(groups.items()):
    baseline='CD'
    if engine==baseline or (case,baseline) not in groups: continue
    ratios=[medians[case,baseline,r]/medians[case,engine,r] for r in range(1,expected_reps+1)]
    comparisons.append(dict(case=case,alternative=engine,median_CD_over_alternative=st.median(ratios),min=min(ratios),max=max(ratios)))

accuracy=rows('accuracy-*.csv')
accuracy_summary=[]
ag=defaultdict(list)
for r in accuracy: ag[r['case'],r['engine']].append(r)
for (case,engine),rs in sorted(ag.items()):
    accuracy_summary.append(dict(case=case,engine=engine,max_relative_prox_residual=max(float(r['max_relative_prox_residual']) for r in rs),max_abs_difference=max(float(r['max_abs_difference']) for r in rs),max_relative_objective_difference=max(float(r['max_relative_objective_difference']) for r in rs)))

mc=rows('mc-*.csv')
mg=defaultdict(list)
lookup={}
for r in mc:
    key=int(r['scenario']),r['method']
    seed=int(r['seed'])
    assert (key[0],key[1],seed) not in lookup, (key,seed)
    lookup[key[0],key[1],seed]=r
    mg[key].append(r)
    V=int(r['false']);R=int(r['selected']);k=int(r['k'])
    assert 0<=V<=R<=int(r['p'])
    ids=list(map(int,filter(None,r['selected_ids'].split(';'))))
    assert len(ids)==len(set(ids))==R
    assert all(1<=j<=int(r['p']) for j in ids) and sum(j>k for j in ids)==V
    assert abs(float(r['fdp'])-V/max(1,R))<1e-12
    if k: assert abs(float(r['power'])-(R-V)/k)<1e-12
mc_summary=[]
expected_methods = {'slope-CD','slope-SLOPE.jl','slope-R','mx-native-equi-Knockoffs.jl','mx-native-mvr-Knockoffs.jl','mx-native-equi-R','cpss-cap10-CD','cpss-cap20-CD','cpss-cap10-R','cpss-cap20-R','cpss-path10-R'}
expected_methods |= {f'mx-{m}-{e}' for m in ('equi','mvr','maxent') for e in ('CD','GLMNet.jl','R')}
for sc in range(1,7):
    wanted=expected_methods|({f'fx-{e}' for e in ('CD','GLMNet.jl','R')} if sc!=6 else set())
    assert {method for scenario,method in mg if scenario==sc}==wanted,sc
for (sc,method),rs in sorted(mg.items()):
    assert len(rs)==expected_reps*expected_b,(sc,method,len(rs))
    fdr,fdrse=stats([float(r['fdp']) for r in rs])
    power,powerse=stats([float(r['power']) for r in rs])
    false,falsese=stats([float(r['false']) for r in rs])
    selection,se=stats([float(r['selected']) for r in rs])
    mc_summary.append(dict(scenario=scenarios[sc-1],method=method,n=len(rs),mean_fdp=fdr,fdp_se=fdrse,power=power,power_se=powerse,mean_false=false,false_se=falsese,mean_selected=selection))

matches=[]
matching=[('slope-CD','slope-SLOPE.jl'),('slope-CD','slope-R'),('cpss-cap10-CD','cpss-cap10-R'),('cpss-cap20-CD','cpss-cap20-R')]
matching += [(f'mx-{m}-CD',f'mx-{m}-{e}') for m in ('equi','mvr','maxent') for e in ('R','GLMNet.jl')]
matching += [('fx-CD',f'fx-{e}') for e in ('R','GLMNet.jl')]
for sc in range(1,7):
    for a,b in matching:
        if (sc,a) not in mg or (sc,b) not in mg: continue
        mismatch=0;maxdiff=0;freq=[]
        for ra in mg[sc,a]:
            rb=lookup[sc,b,int(ra['seed'])]
            sa=set(filter(None,ra['selected_ids'].split(';')));sb=set(filter(None,rb['selected_ids'].split(';')))
            mismatch+=sa!=sb;maxdiff=max(maxdiff,len(sa^sb))
        matches.append(dict(scenario=scenarios[sc-1],reference=a,alternative=b,n=len(mg[sc,a]),selection_mismatches=mismatch,max_symmetric_difference=maxdiff))

power_differences=[]
for sc in (1,2,3,4,6):
    for m in ('mvr','maxent'):
        diffs=[float(r['power'])-float(lookup[sc,'mx-equi-CD',int(r['seed'])]['power']) for r in mg[sc,f'mx-{m}-CD']]
        mean,se=stats(diffs)
        power_differences.append(dict(scenario=scenarios[sc-1],comparison=f'{m} minus equi',paired_power_difference=mean,mc_se=se))

validity=rows('validity-*.csv')
vg=defaultdict(list)
for r in validity: vg[r['engine']].append(r)
validity_summary=[]
for engine,rs in sorted(vg.items()):
    vals=[float(r['relative_covariance_error']) for r in rs]
    implied=[float(r['implied_covariance_error']) for r in rs if r.get('implied_covariance_error')]
    validity_summary.append(dict(engine=engine,replications=len(rs),median_empirical_relative_covariance_error=st.median(vals),min=min(vals),max=max(vals),max_implied_covariance_error=max(implied) if implied else None))
fixed_validity=rows('fixed-validity-*.csv')
cpss_agreement=rows('cpss-agreement-*.csv')

def save(name,data):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

for name,data in [('timing-summary.csv',timing_summary),('paired-time-ratios.csv',comparisons),('accuracy-summary.csv',accuracy_summary),('simulation-summary.csv',mc_summary),('selection-agreement.csv',matches),('paired-power-differences.csv',power_differences),('constructor-validity.csv',validity_summary)]: save(name,data)

with (out/'summary.md').open('w') as f:
    f.write('# Direct comparison on Discovery\n\n')
    f.write(f'{expected_reps} independent process replications; {expected_reps*expected_b} shared datasets per scenario. Timings exclude startup/JIT and accuracy checks.\n\n')
    f.write('| Case | Engine | Median ms | Replication min–max ms |\n|---|---|---:|---:|\n')
    for r in timing_summary: f.write(f"| {r['case']} | {r['engine']} | {r['median_ms']:.3f} | {r['min_ms']:.3f}–{r['max_ms']:.3f} |\n")
    f.write('\nCD/alternative ratios greater than one mean CD is slower.\n\n| Case | Alternative | Median paired CD/alternative | Min–max |\n|---|---|---:|---:|\n')
    for r in comparisons: f.write(f"| {r['case']} | {r['alternative']} | {r['median_CD_over_alternative']:.2f} | {r['min']:.2f}–{r['max']:.2f} |\n")
    f.write('\n## Common accuracy certificate\n\n| Case | Engine | Maximum relative prox residual | Maximum coefficient difference vs CD | Maximum relative objective difference |\n|---|---|---:|---:|---:|\n')
    for r in accuracy_summary: f.write(f"| {r['case']} | {r['engine']} | {r['max_relative_prox_residual']:.3g} | {r['max_abs_difference']:.3g} | {r['max_relative_objective_difference']:.3g} |\n")
    f.write('\n## Constructor covariance check\n\nEmpirical check: 20,000 independent conditional draws at X=0,p=30,AR(.8), repeated in each process. Implied covariance is recovered using the shared standard-normal innovations.\n\n')
    f.write('| Engine | Empirical relative error, median [range] | Maximum implied covariance error |\n|---|---:|---:|\n')
    for r in validity_summary: f.write(f"| {r['engine']} | {r['median_empirical_relative_covariance_error']:.4f} [{r['min']:.4f}, {r['max']:.4f}] | {r['max_implied_covariance_error']} |\n")
    f.write('\n## Shared-design simulations\n\nKnown Gaussian population covariance; q=.1 for SLOPE/knockoff+. CPSS targets its documented conditional PFER bound, not FDR=.1. Native constructor pipelines use language-specific random draws. Refer to constructor checks before interpreting their error-control results.\n\n| Scenario | Method | Mean FDP (MC SE) | Power (MC SE) | Mean false selections |\n|---|---|---:|---:|---:|\n')
    for r in mc_summary: f.write(f"| {r['scenario']} | {r['method']} | {r['mean_fdp']:.4f} ({r['fdp_se']:.4f}) | {r['power']:.4f} ({r['power_se']:.4f}) | {r['mean_false']:.3f} |\n")
    f.write('\n## Matched implementation agreement\n\n| Scenario | Reference | Alternative | Mismatched selection sets / datasets | Maximum set difference |\n|---|---|---|---:|---:|\n')
    for r in matches: f.write(f"| {r['scenario']} | {r['reference']} | {r['alternative']} | {r['selection_mismatches']}/{r['n']} | {r['max_symmetric_difference']} |\n")
    f.write('\n## Paired power differences from construction choices\n\nShared data, Gaussian innovations, conditional sampler, lasso solver and lambda. Only the diagonal construction differs.\n\n| Scenario | Choice | Power difference (MC SE) |\n|---|---|---:|\n')
    for r in power_differences: f.write(f"| {r['scenario']} | {r['comparison']} | {r['paired_power_difference']:.4f} ({r['mc_se']:.4f}) |\n")

receipt={'replications':expected_reps,'datasets_per_scenario':expected_reps*expected_b,'timing_rows':len(timings),'mc_rows':len(mc),'accuracy_rows':len(accuracy),'maximum_relative_prox_residual':max(r['max_relative_prox_residual'] for r in accuracy_summary),'matching_comparisons':len(matches),'selection_mismatches':sum(r['selection_mismatches'] for r in matches),'max_fixed_gram_error':max((float(r['max_gram_error']) for r in fixed_validity),default=None),'max_fixed_cross_gram_error':max((float(r['max_cross_gram_error']) for r in fixed_validity),default=None),'max_cpss_frequency_difference':max((float(r['max_frequency_difference']) for r in cpss_agreement),default=None),'status':'coverage_and_arithmetic_checked'}
(out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
