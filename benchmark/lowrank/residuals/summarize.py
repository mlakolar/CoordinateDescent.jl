import csv, json, pathlib, random, statistics, sys, tomllib
from common import CASES
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];base=root/'results'/phase
reps=[0] if phase=='pilot' else list(range(1,21))
rows=[];components=[];metadata=[];resources=[]
for p in sorted(base.glob('*-*/*/timings.csv')):rows.extend(csv.DictReader(p.open()))
for p in sorted(base.glob('*-*/*/components.csv')):components.extend(csv.DictReader(p.open()))
for label in [f'{c}-{r}' for c in CASES for r in reps]:
    for variant in ('baseline','candidate','python','r'):
        p=base/label/variant
        try:
            if variant in ('baseline','candidate'):md=tomllib.loads((p/'metadata.toml').read_text())
            elif variant=='python':md=json.loads((p/'metadata.json').read_text())
            else:md=next(csv.DictReader((p/'metadata.csv').open()))
            metadata.append(md)
            rd=json.loads((p/'resource.json').read_text());rd.update(case=label.rsplit('-',1)[0],rep=int(label.rsplit('-',1)[1]),variant=variant)
            resources.append(rd)
        except Exception:pass
validation=json.loads((base/'validation.json').read_text())
expected=42*len(CASES)*len(reps)
index={}
for r in rows:index.setdefault((r['case'],r['variant'],r['mode'],int(r['rep'])),[]).append(r)
table=[]
for case,variant,mode in sorted(set((r['case'],r['variant'],r['mode']) for r in rows)):
    values=[];allocated=[]
    for rep in reps:
        z=index.get((case,variant,mode,rep),[])
        if len(z)==3:
            values.append(statistics.median(float(x['seconds']) for x in z))
            if z[0]['bytes']:allocated.append(statistics.median(int(x['bytes']) for x in z))
    if values:table.append(dict(case=case,variant=variant,mode=mode,independent_inputs=len(values),median_ms=1000*statistics.median(values),
        min_input_ms=1000*min(values),max_input_ms=1000*max(values),median_allocated_bytes=statistics.median(allocated) if allocated else None))
rng=random.Random(10126026);comparisons=[]
comparators=[('candidate','baseline',m) for m in ('public_fit','public_path','coefficient_fit','coefficient_path')]
comparators += [(j,'python',m) for j in ('candidate','baseline') for m in ('public_fit','public_path','coefficient_fit','coefficient_path')]
comparators += [('candidate','r',m) for m in ('public_fit','public_path')]
for case in CASES:
    for numerator,denominator,mode in comparators:
        ratios=[]
        for rep in reps:
            a=index.get((case,numerator,mode,rep),[]);b=index.get((case,denominator,mode,rep),[])
            if len(a)==len(b)==3:ratios.append(statistics.median(float(x['seconds']) for x in a)/statistics.median(float(x['seconds']) for x in b))
        if ratios:
            boots=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
            comparisons.append(dict(case=case,numerator=numerator,denominator=denominator,mode=mode,n=len(ratios),paired_median_ratio=statistics.median(ratios),
                min_ratio=min(ratios),max_ratio=max(ratios),ci_low=boots[249],ci_high=boots[9749],ratios=ratios))
component_table=[]
for case,variant,name in sorted(set((r['case'],r['variant'],r['component']) for r in components)):
    values=[]
    for rep in reps:
        z=[r for r in components if r['case']==case and r['variant']==variant and r['component']==name and int(r['rep'])==rep]
        if len(z)==3:values.append(statistics.median(float(r['seconds']) for r in z))
    if values:component_table.append(dict(case=case,variant=variant,component=name,n=len(values),median_ms=1000*statistics.median(values)))
memory=[]
for case in CASES:
    for variant in ('baseline','candidate','python','r'):
        z=[m for m in metadata if m['case']==case and m['variant']==variant]
        if not z:continue
        memory.append(dict(case=case,variant=variant,n=len(z),warmed_current_mib=statistics.median(float(x['warmed_current_rss']) for x in z)/2**20,
            warmed_peak_mib=statistics.median(float(x['warmed_peak_rss']) for x in z)/2**20,
            timing_peak_mib=statistics.median(float(x['peak_after_timing']) for x in z)/2**20,
            whole_worker_peak_mib=statistics.median(float(x['whole_worker_peak_rss']) for x in z)/2**20,
            max_whole_worker_mib=max(float(x['whole_worker_peak_rss']) for x in z)/2**20))
for name,data in [('all-timings.csv',rows),('aggregate.csv',table),('comparisons.csv',[{k:v for k,v in r.items() if k!='ratios'} for r in comparisons]),('components.csv',component_table),('memory.csv',memory)]:
    if data:
        with (base/name).open('w') as f:
            w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
summary=dict(phase=phase,timing_rows=len(rows),expected_timing_rows=expected,component_rows=len(components),validation=validation,
    profile_reports=len(list(base.glob('*-*/*/cpu-*.txt'))),allocation_reports=len(list(base.glob('*-*/*/alloc-*.txt'))),
    resource_records=len(resources),failed_processes=[r for r in resources if r['exit_code']!=0],table=table,comparisons=comparisons,memory=memory,component_table=component_table)
(base/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Fixed-rank residual construction: Discovery study','',
       f'Phase: {phase}. {len(reps)} independent shared inputs in each of six designs, three warmed rounds per method/input. The seed-zero pilot is separate from evaluation inputs1–20. No policy is retuned on evaluation outcomes.','',
       'Complete native public fits/paths are separate from coefficient-only controls. The SciPy full-output adapter returns equivalent coefficients, residuals, intercepts, rank, support, objective and both DF definitions, using thin factors with the same conservative dense rank fallback. R is rrpack\'s native QR/SVD interface; its output contract differs and its wide/rank-deficient coefficients need not be minimum norm. Invalid R normal-equation shortcuts are excluded.','',
       '## Public API comparison','',
       '| Case | Mode | Baseline Julia ms | Candidate Julia ms | Candidate/baseline (95% interval) | SciPy full-output ms | R QR/SVD ms |','|---|---|---:|---:|---|---:|---:|']
def lookup(case,variant,mode):return next(x for x in table if (x['case'],x['variant'],x['mode'])==(case,variant,mode))
for case in CASES:
    for mode in ('public_fit','public_path'):
        try:
            a=lookup(case,'baseline',mode);b=lookup(case,'candidate',mode);py=lookup(case,'python',mode);rr=lookup(case,'r',mode)
            ratio=next(x for x in comparisons if (x['case'],x['numerator'],x['denominator'],x['mode'])==(case,'candidate','baseline',mode))
            lines.append(f"| {case} | {mode} | {a['median_ms']:.3f} | {b['median_ms']:.3f} | {ratio['paired_median_ratio']:.3f} ({ratio['ci_low']:.3f}–{ratio['ci_high']:.3f}) | {py['median_ms']:.3f} | {rr['median_ms']:.3f} |")
        except StopIteration:lines.append(f'| {case} | {mode} | incomplete | | | | |')
lines += ['', 'Times summarize within-input medians. Ratios use paired input medians; intervals bootstrap the median of independent input ratios (10,000 resamples). Ranges and every ratio are retained. These are warm-process, newly constructed fits, not process-startup latencies. Imports/JIT/loading/checks/profiles are outside timers; centering, decomposition and required outputs are inside public timers.','',
    '## Coefficient-only controls','', '| Case | Mode | Julia baseline ms | Julia candidate ms | SciPy ms |','|---|---|---:|---:|---:|']
for case in CASES:
    for mode in ('coefficient_fit','coefficient_path'):
        try:
            a,b,py=[lookup(case,v,mode) for v in ('baseline','candidate','python')]
            lines.append(f"| {case} | {mode} | {a['median_ms']:.3f} | {b['median_ms']:.3f} | {py['median_ms']:.3f} |")
        except StopIteration:pass
lines += ['', 'Cached-input component timings are in components.csv. They are not an additive decomposition of public runtime. CPU profiles cover every independent many-response/wide input for baseline, candidate, SciPy and R; allocation profiles are separate from timing. Profile inclusive frames are not additive wall-time percentages.','',
    '## Correctness and memory','', '```json',json.dumps(validation,indent=2),'```','',
    'Independent certificates check closed-form loss, feasible rank, predictions, minimum-norm representatives where required, and all native/SciPy returned residual/intercept/support/DF fields. R nonunique coefficients are judged through identifiable fitted values; coefficient/minimum-norm differences remain in certificates.csv. Every baseline/candidate coefficient pair is checked directly.','',
    'memory.csv records fresh process per input and implementation, current/peak RSS after warming, peak after timing and whole-worker peak. Workers cover the four modes (two for R); RSS includes imports/JIT/data/output persistence and whole-worker peaks also include separate components/profiles. It is not optimizer workspace or a single-fit memory bound. Julia cumulative allocated bytes are reported separately in aggregate.csv.','',
    '## Interpretation and recreation','',
    'Do not adopt the candidate from inner-kernel savings alone. Judge public fit/path paired ratios and their uncertainty, preserve small or reversed effects and any remaining SciPy gap, and retain the existing design-rank cutoff and diagnostic fallbacks. The predeclared multiplication choice is rank*(p+q)<p*q; this study changes no statistical objective, rank-selection rule or tolerance. A default recommendation requires complete validated results.','',
    'Pinned baseline:295fbf5420d6384a1ca261023367ee86d0a40b8c. Candidate production:9dd42f86f4a89c947b8205407de065cf12b9fbbf. Frozen PB/PS/KS and Julia environments are recorded in the run registry and worker metadata. Timing Julia1.13.2, Discovery EPYC7513, one Julia/BLAS/OpenMP/MKL thread. All numerical generation, fitting, profiling, validation and aggregation run under Slurm. Eight exclusive workers within8CPUs/32GiB. See README and executable launchers; no merges are authorized.','']
(base/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in ('table','comparisons','memory','component_table')},indent=2))
assert len(rows)==expected and not summary['failed_processes'] and validation['failed_certificates']==validation['failed_pairs']==0 and not validation['errors']
