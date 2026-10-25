"""Summarize completed holdout records under Slurm; never refit."""
import csv,hashlib,json,pathlib,random,re,statistics,sys,tomllib
root=pathlib.Path(sys.argv[1]);base=root/'results/main'
validation=json.loads((base/'inspection.json').read_text())
timings=[];profiles=[];memories=[];processes=[];errors=[]
expected_sources={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in
    [('baseline',root.parent/'issue99/baseline/src/structured.jl'),('candidate',root/'candidate3/src/structured.jl')]}
for folder in sorted(base.glob('*-*')):
    if not folder.is_dir():continue
    for variant in ('baseline','candidate','reference'):
        dest=folder/variant
        try:
            resource=json.loads((dest/'resource.json').read_text());processes.append(dict(input=folder.name,variant=variant,**resource))
            if resource['exit_code']!=0:errors.append(dict(input=folder.name,variant=variant,error='failed process',outcome=resource))
            if variant=='reference':continue
            ts=list(csv.DictReader((dest/'timings.csv').open()));timings.extend(ts)
            md=tomllib.loads((dest/'metadata.toml').read_text())
            assert md['source_sha256']==expected_sources[variant]
            assert md['cpu']=='AMD EPYC 7513 32-Core Processor' and md['julia_threads']==md['blas_threads']==1
            memories.append(dict(input=folder.name,variant=variant,**{k:md[k] for k in
                ('warmed_current_rss','warmed_peak_rss','current_after_timing','peak_after_timing','whole_worker_peak_rss')}))
            text=(dest/'cpu.txt').read_text();snapshots=re.search(r'Total snapshots: (\d+)',text)
            alloc=(dest/'alloc.txt').read_text();samples=re.search(r'sampled_allocations=(\d+)',alloc)
            profiles.append(dict(input=folder.name,variant=variant,snapshots=int(snapshots.group(1)) if snapshots else None,
                interval=md['cpu_profile_delay'],allocation_samples=int(samples.group(1)) if samples else None))
        except Exception as exc:errors.append(dict(input=folder.name,variant=variant,error=repr(exc)))
cases=('small','irregular','linear100k','quadratic100k','tvpath','linearpath','quadraticpath','linear1m','quadratic1m')
table=[];rng=random.Random(98261026)
for case in cases:
    ratios=[];before=[];after=[];ba=[];ca=[];bi=[];ci=[]
    for rep in (1,2,3):
        values={v:[r for r in timings if r['case']==case and r['variant']==v and int(r['rep'])==rep] for v in ('baseline','candidate')}
        if all(len(z)==3 for z in values.values()):
            b,c=[statistics.median(float(r['seconds']) for r in values[v]) for v in ('baseline','candidate')]
            before.append(b);after.append(c);ratios.append(c/b)
            ba.append(statistics.median(int(r['bytes']) for r in values['baseline']));ca.append(statistics.median(int(r['bytes']) for r in values['candidate']))
            bi.append(int(values['baseline'][0]['iterations']));ci.append(int(values['candidate'][0]['iterations']))
    if ratios:
        boots=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
        table.append(dict(case=case,n=len(ratios),baseline_seconds=statistics.median(before),candidate_seconds=statistics.median(after),
            paired_ratio=statistics.median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),ci_low=boots[249],ci_high=boots[9749],ratios=ratios,
            baseline_allocated_bytes=statistics.median(ba),candidate_allocated_bytes=statistics.median(ca),baseline_iterations=bi,candidate_iterations=ci))
for name,rows in [('all-timings.csv',timings),('memory.csv',memories),('profiles.csv',profiles),('processes.csv',processes)]:
    if rows:
        with (base/name).open('w') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(validation=validation,errors=errors,timing_rows=len(timings),expected_timing_rows=162,source_hashes=expected_sources,
    process_records=len(processes),profile_records=len(profiles),table=table)
(base/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Trend product holdout study','',
    'Three fresh independent shared inputs per case, three warmed complete public fits/paths per input and implementation. Paired before/after comparisons use within-input medians, with method order rotated. The seed-zero calibration is excluded. All numerical work ran under Discovery Slurm on one fixed EPYC7513 allocation, with one Julia/BLAS/OpenMP thread per exclusive worker.','',
    '| Case | Inputs | Baseline s | Candidate s | Paired median | Input ratio range |','|---|---:|---:|---:|---:|---|']
for r in table:lines.append(f"| {r['case']} | {r['n']} | {r['baseline_seconds']:.6f} | {r['candidate_seconds']:.6f} | {r['paired_ratio']:.3f} | {r['min_ratio']:.3f}–{r['max_ratio']:.3f} |")
lines += ['', 'Every input ratio and descriptive bootstrap interval is in summary.json. Three independent inputs provide limited precision; timing rounds are not additional independent replications. Preserve small/reversed effects and all controls. Public timers include setup, factorization, iterations, results and DF. Imports, input loading, compilation/warming, independent checks and profiles are outside timers.','',
    'The independent glmgen fits use obj_tol=1e-12, thinning disabled and the same original input bytes/grid. They provide cross-solver agreement checks, not a certified primal-dual gap or a timed performance ranking. Coefficient agreement must be<=1e-4 and objective difference<=1e-6 for every returned level. The candidate changes product kernels only; rho, TV prox, banded factors and stopping/DF tolerances are fixed.','',
    '## Validation','', '```json',json.dumps({k:v for k,v in validation.items() if k not in ('results','native_records','reference_records')},indent=2),'```','',
    'Full per-input native and reference checks are in inspection.json. Process errors, incomplete coverage and native nonconvergence remain visible and preclude an adoption claim. Allocation totals are cumulative fit/path bytes. Memory records distinguish warmed current/peak, timing peak and final whole-worker peak, which includes separate profile work; none is isolated solver workspace. Profiles cover every dataset separately and record snapshots, intervals and sampled allocations. Inclusive frames overlap.','',
    '## Provenance and decision','',
    'Baseline295fbf5 and the frozen candidate use the same pinned PB/PS/KS dependencies and Julia1.13.2. Source hashes, CPU/thread metadata, saved seeds/input hashes, process outcomes, R sessions and exact launchers are retained. Raw input/fitted arrays and source/environment snapshots remain in the owning run. See README and run registry. No merge is authorized. A recommendation requires complete validated coverage and repeatable useful public gains; inspect the final ratios and adverse evidence before changing the internal default.','']
(base/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in ('validation','table')},indent=2))
assert not errors and not validation['errors'] and len(timings)==162 and len(processes)==81 and len(profiles)==54
