"""Read completed frozen cohorts only. No generation, optimization or refitting."""
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tomllib
import numpy as np
from study_check import fits

BASE=Path(os.environ['HDM_BASE'])
STAT=BASE/'stat-main-v6';RUN=BASE/'runtime-main-v6';OUT=BASE/'completion-report-v2'
OUT.mkdir(exist_ok=False)

def read_csv(path):return list(csv.DictReader(path.open()))
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(value):
    if value in ('',None):return 'undefined'
    return f'{float(value):.4f}'
def ci(r,key):return f"{fmt(r[key+'_mean'])} [{fmt(r[key+'_lo'])}, {fmt(r[key+'_hi'])}]"

# All native statistical results must meet the declared independent criterion.
failures=defaultdict(list);counts=Counter();native={};nonconvergence=[]
for path in sorted((STAT/'checks').glob('*.csv')):
    if '.agreement.' in path.name:continue
    for r in read_csv(path):
        group=(r['engine'],r['method'],int(r['scenario']))
        counts[group]+=1
        if r['valid']!='True':failures[group].append(r)
        if r['engine']=='native':native[int(r['seed']),r['method']]=r
accuracy=[]
for (engine,method,sc),count in sorted(counts.items()):
    failed=failures[engine,method,sc]
    accuracy.append(dict(engine=engine,method=method,scenario=sc,total=count,failed=len(failed),
       max_failed_residual=max((float(r['residual']) for r in failed),default=0)))
assert sum(counts.values())==177408
assert sum(r['total'] for r in accuracy if r['engine']=='native')==64512
assert all(not r['failed'] for r in accuracy if r['engine']=='native')
write_csv(OUT/'statistical-accuracy.csv',accuracy)
for p in sorted((STAT/'fits').glob('native*.csv')):
    nonconvergence.extend(dict(file=p.name,**r) for r in read_csv(p) if r['reported_converged']=='false')
write_csv(OUT/'native-strict-stopping.csv',nonconvergence)
assert len(nonconvergence)==5

# Check emitted source/CPU/thread metadata from every Julia worker, not a sample.
provenance=[]
for cohort in (STAT,RUN):
    for p in sorted((cohort/'fits').glob('*.toml')):
        d=tomllib.loads(p.read_text());assert d['julia_threads']==d['blas_threads']==1
        assert '7513' in d['cpu'] and d['SLOPE']=='1.3.1' and d['julia']=='1.10.12'
        assert d['hdm_slope_sha256']=='3f0574d288186568a553150e28f5ed1727add4dab0f508593fb2cd22de036dc3'
        assert d['sj_regweights_sha256']=='15869777730d2dc902db2e19277969c889c183943eca60071a2fd993bfb1e206'
        provenance.append(dict(cohort=cohort.name,file=p.name,**d))
    for p in sorted((cohort/'fits').glob('*.session.txt')):
        assert 'SLOPE_2.1.1' in p.read_text() and 'R version 4.5.3' in p.read_text()
write_csv(OUT/'julia-provenance.csv',provenance)
assert len(provenance)==224+56

# Actual constructor conventions and selected sets, using saved W/B only.
conventions=[]
for sc in range(1,17):
    rows=[]
    for shard in range(1,8):
        ni=fits(STAT/'fits'/f'native-sc{sc}-shard{shard}.bin')
        ri=fits(STAT/'fits'/f'R-sc{sc}-shard{shard}.bin')
        for (seed,W,B),(rseed,RW,RB) in zip(ni,ri,strict=True):
            assert seed==rseed
            rows.append((np.max(np.abs(W[:,1]-W[:,2]))/np.max(W[:,1]),
                         np.max(np.abs(W[:,1]-RW[:,3]))/np.max(W[:,1]),
                         np.count_nonzero((np.abs(B[:-1,1])>1e-6)!=(np.abs(RB[:-1,3])>1e-6))))
    conventions.append(dict(scenario=sc,inputs=len(rows),max_relative_paper_Julia_weight_difference=max(r[0] for r in rows),
       max_relative_paper_Rnative_weight_difference=max(r[1] for r in rows),Rnative_paper_selection_disagreements=int(sum(r[2]>0 for r in rows))))
write_csv(OUT/'actual-calibrations.csv',conventions)

# Runtime objective/selection matching uses certificates already computed on X/y.
runtime_rows=[];by_point={};by_group=defaultdict(list);rounds=defaultdict(list)
for p in sorted((RUN/'checks').glob('*.csv')):
    for r in read_csv(p):
        assert r['valid']=='True'
        key=(r['seed'],r['method'],r['payload'],r['round'],r['point'])
        by_point[key,r['engine'],r['setting']]=r
        by_group[r['engine'],r['setting']].append(r)
agreements=[]
for (key,engine,setting),a in by_point.items():
    if engine=='native' and setting=='public':continue
    b=by_point[key,'native','public']
    agreements.append(dict(engine=engine,setting=setting,relative_objective=abs(float(a['objective'])-float(b['objective']))/max(1.,abs(float(b['objective']))),
       selected_set_disagreement=a['selected_ids']!=b['selected_ids']))
agreement_summary=[]
for engine,setting in sorted({(r['engine'],r['setting']) for r in agreements}):
    rs=[r for r in agreements if (r['engine'],r['setting'])==(engine,setting)]
    agreement_summary.append(dict(engine=engine,setting=setting,point_pairs=len(rs),max_relative_objective=max(r['relative_objective'] for r in rs),
       selected_set_disagreements=sum(r['selected_set_disagreement'] for r in rs)))
write_csv(OUT/'runtime-objective-selection.csv',agreement_summary)

# All timed allocation rows, resource receipts and default comparisons.
timings=read_csv(RUN/'report'/'timings.csv');heap=defaultdict(list)
for r in timings:
    if r['payload']!='weights' and r['bytes'] not in ('NA',''):
        heap[int(r['case']),r['engine'],r['method'],r['payload'],r['setting']].append(float(r['bytes']))
write_csv(OUT/'timed-heap.csv',[dict(case=k[0],engine=k[1],method=k[2],payload=k[3],setting=k[4],rows=len(v),median_bytes=float(np.median(v)),
       minimum_bytes=min(v),maximum_bytes=max(v)) for k,v in sorted(heap.items())])
resources=[]
for p in sorted((RUN/'logs').glob('*.resource.json')):
    r=json.loads(p.read_text());assert r['returncode']==0
    engine,sc,rep=p.name.split('.')[0].split('-')
    resources.append(dict(engine=engine,case=int(sc),rep=int(rep),**r))
assert len(resources)==84
write_csv(OUT/'whole-worker-resources.csv',resources)
rss=[]
for sc in range(1,5):
    for engine in ('native','julia','R'):
        values=[r['max_rss_bytes'] for r in resources if (r['case'],r['engine'])==(sc,engine)]
        rss.append(dict(case=sc,engine=engine,processes=len(values),median_rss_bytes=float(np.median(values)),min_rss_bytes=min(values),max_rss_bytes=max(values)))
write_csv(OUT/'whole-worker-rss.csv',rss)

mc=read_csv(STAT/'report'/'monte-carlo.csv');paired=read_csv(STAT/'report'/'paired-differences.csv')
selected_mc=[r for r in mc if r['engine']=='native' and r['method'] in ('bh','paper')]
assert all(r['valid']=='1008' and r['failed']=='0' for r in selected_mc)
write_csv(OUT/'native-primary-monte-carlo.csv',selected_mc)
ratios=read_csv(RUN/'report'/'paired-runtime.csv')
assert all(r['independent_processes']=='7' and r['inputs']=='21' for r in ratios)
stat_summary=json.loads((STAT/'report'/'summary.json').read_text())
runtime_summary=json.loads((RUN/'report'/'summary.json').read_text())
summary=dict(statistical_inputs=16128,main_inputs=13104,stress_inputs=3024,native_statistical_certificates=64512,
 native_failed_statistical_certificates=0,external_failed_statistical_certificates=sum(r['failed'] for r in accuracy),
 native_strict_stopping_flags=len(nonconvergence),runtime_inputs=84,runtime_certificates=94500,runtime_failures=0,
 runtime_complete_field_allocation_witnesses=504,Julia_source_CPU_thread_receipts=len(provenance),runtime_fresh_processes=len(resources),
 runtime_cpu_profiles=runtime_summary['cpu_profiles'],runtime_allocation_profiles=runtime_summary['allocation_profiles'],
 runtime_objective_selection=agreement_summary,actual_weight_conventions=conventions,
 note='No fits/generation/tolerance changes. All original external failures and strict native flags retained. Statistical and runtime cohorts analyzed separately; some PRNG seed numbers overlap across these differently sized cohorts.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

lines=['# SLOPE completion tables from frozen cohorts','',
 'All numerical generation, fitting, independent checks, profiling and aggregation ran under Discovery Slurm.',
 'Native statistical fits:64512/64512 independent original-design residual certificates pass on16128 independent inputs. '
 'Five strict1e-9 stopping flags are retained; each still passes the declared1e-7 independent criterion. '
 '450 external statistical certificates fail at their frozen controls and remain excluded from valid-backend rankings. '
 'All94500 runtime/default certificates pass. No failed result is relabeled as certified.','',
 'FDR is meanFDP; power is the fraction of20 true coefficients selected (undefined at global null). '
 'Pointwise95% Student-t Monte Carlo intervals are shown. Complete data include HoeffdingFDP bounds and exact-null binomial intervals. '
 'Every scenario has1008 inputs; main and stress cohorts remain separate.','',
 '| Scenario | BH FDR [CI] | Gaussian FDR [CI] | BH power [CI] | Gaussian power [CI] | Paired FDR change [CI] | Paired power change [CI] |',
 '|---|---|---|---|---|---|---|']
for sc in range(1,17):
    bh=next(r for r in selected_mc if int(r['scenario'])==sc and r['method']=='bh')
    ga=next(r for r in selected_mc if int(r['scenario'])==sc and r['method']=='paper')
    d=next(r for r in paired if int(r['scenario'])==sc and r['engine']=='native' and r['method']=='paper')
    lines.append(f"| {bh['name']} ({bh['cohort']}) | {ci(bh,'fdp')} | {ci(ga,'fdp')} | {ci(bh,'power')} | {ci(ga,'power')} | {ci(d,'fdp_difference')} | {ci(d,'power_difference')} |")
lines+=['','## Complete public procedure runtime', '',
 'Comparator/native public geometric ratios, with pointwise process-paired95% intervals. '
 'Each case uses21 inputs over7 fresh process replications, three warmed rounds/input. '
 'Complete fit/path results are timed; preprocessing/profile/certificate placement is documented in PROTOCOL. '
 'These are current-procedure comparisons, not a before/after speed claim for the new calibration API.','',
 '| Case | Weights | Payload | SLOPE.jl/native [CI] | R/native [CI] | Prepared/native [CI] | Default/native [CI] | Unscreened/native [CI] |',
 '|---|---|---|---|---|---|---|---|']
def ratio(sc,method,payload,engine,setting):
    rs=[r for r in ratios if (int(r['case']),r['method'],r['payload'],r['engine'],r['setting'])==(sc,method,payload,engine,setting)]
    if not rs:return 'not applicable'
    r=rs[0];return f"{fmt(r['geometric_ratio'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]"
for sc,name in enumerate(('dense','wide','correlated','sparse'),1):
    for method in ('bh','paper','julia'):
        for payload in ('fit','path'):
            lines.append(f"| {name} | {method} | {payload} | {ratio(sc,method,payload,'julia','public')} | {ratio(sc,method,payload,'R','public')} | {ratio(sc,method,payload,'native','prepared')} | {ratio(sc,method,payload,'native','default')} | {ratio(sc,method,payload,'native','unscreened')} |")
lines+=['','## Scope and retained limits','',
 'No empirical FDR<=q acceptance rule applies outside the known-noise orthogonal theorem. '
 'Gaussian adjustment, supplied std(y), correlated/misspecified covariance and selected path levels do not inherit that guarantee. '
 'The native Gaussian rule has a declared first-nondecrease tail and differs from the pinned Julia convention. '
 'Statistical primary fits use native1e-9 and external1e-8; runtime uses native1e-8 and external1e-10, all against the same independent1e-7 criterion. '
 'Production defaults remain BH, tol1e-7/max10000 and screening=true. '
 'Whole-worker RSS includes imports/JIT/profiling, so it is not fitted-state memory. '
 'Pilots, failed launchers, external accuracy failures and native strict-stopping flags remain in the evidence. '
 'Within each cohort inputs are independent. Runtime seed numbers overlap some statistical scenario seeds with different dimensions/data laws; '
 'the cohorts are not pooled into a combined independent sample.','']
(OUT/'TABLES.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
