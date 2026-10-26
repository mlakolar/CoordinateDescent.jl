"""Read saved study outputs under Slurm; never fit or alter primary receipts."""
import argparse, csv, hashlib, json, math, pathlib, re, statistics, tomllib
import numpy as np
from scipy.stats import t

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='results/study-v1')
parser.add_argument('--out', default='results/study-report-v1')
parser.add_argument('--canary', action='store_true')
args = parser.parse_args()
root, out = pathlib.Path(args.root), pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
s = json.loads((root / 'summary.json').read_text())
assert s['successful_processes'] + len(s['failures']) == s['expected_processes']
complete_ids = {(p['case'],p['rep'],p['method']) for p in s['processes']}
medians = [r for r in s['medians'] if (r['case'],r['rep'],r['method']) in complete_ids]
lookup = {(r['case'], r['rep'], r['method'], r['mode'], r['kind'], r['fraction']): r for r in medians}

def write_csv(name, rows):
    if not rows:
        return
    with (out / (name + '.csv')).open('w') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def interval(ratios):
    if len(ratios) < 2:
        return [None, None]
    z = np.log(ratios)
    radius = float(t.ppf(.975, len(z)-1) * np.std(z, ddof=1) / math.sqrt(len(z)))
    mean = float(np.mean(z))
    return [math.exp(mean-radius), math.exp(mean+radius)]

comparisons = [('candidate', 'baseline'), ('candidate', 'apg'), ('candidate', 'po95'),
               ('candidate', 'py_fista'), ('candidate', 'py_admm'),
               ('pb95', 'baseline'), ('guarded', 'pb95'), ('finalreuse', 'pb95'),
               ('guarded_final', 'pb95'), ('candidate', 'guarded_final'),
               ('quadtrait', 'pb95'), ('pb99_safe', 'pb95'),
               ('cadence1', 'pb95'), ('cadence20', 'pb95'), ('coeffonly', 'pb95')]
pairs, grouped = [], {}
for r in medians:
    for numerator, denominator in comparisons:
        if r['method'] != numerator:
            continue
        b = lookup.get((r['case'], r['rep'], denominator, r['mode'], r['kind'], r['fraction']))
        if b is None:
            continue
        passed = r['all_accuracy_pass'] and b['all_accuracy_pass']
        pair = dict(case=r['case'], rep=r['rep'], numerator=numerator, denominator=denominator,
                    mode=r['mode'], kind=r['kind'], fraction=r['fraction'],
                    accuracy_matched=passed, numerator_seconds=r['seconds'], denominator_seconds=b['seconds'],
                    ratio=r['seconds']/b['seconds'], numerator_allocated=r['allocated'], denominator_allocated=b['allocated'])
        pairs.append(pair)
        if passed:
            key = (r['case'], numerator, denominator, r['mode'], r['kind'], r['fraction'])
            grouped.setdefault(key, []).append(pair)
groups = []
for key, rs in sorted(grouped.items()):
    ratios = [r['ratio'] for r in rs]
    ci = interval(ratios)
    groups.append(dict(zip(['case', 'numerator', 'denominator', 'mode', 'kind', 'fraction'], key),
                       inputs=len(rs), numerator_seconds=statistics.median(r['numerator_seconds'] for r in rs),
                       denominator_seconds=statistics.median(r['denominator_seconds'] for r in rs),
                       median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
                       geomean_ratio=math.exp(statistics.mean(math.log(z) for z in ratios)),
                       geomean_95_low=ci[0], geomean_95_high=ci[1],
                       inputs_slower=sum(z > 1 for z in ratios)))

provenance, profiles, memory, mode_checks, allocation_witnesses, counts, components = [], [], [], [], [], [], []
for p in s['processes']:
    case, rep, method, md = p['case'], p['rep'], p['method'], p['metadata']
    f = root / case / (method if args.canary else 'rep'+str(rep))
    if not args.canary:
        f = f / method
    identity = dict(case=case, rep=rep, method=method)
    py = method.startswith('py_')
    frozen = pathlib.Path('candidate2' if method == 'candidate' else 'baseline')
    if not py:
        assert md['extension_sha256'] == hashlib.sha256((frozen/'ext/HighDimensionalModelsProximalAlgorithmsExt.jl').read_bytes()).hexdigest()
        assert md['lowrank_sha256'] == hashlib.sha256((frozen/'src/lowrank.jl').read_bytes()).hexdigest()
        assert md['worker_sha256'] == hashlib.sha256(pathlib.Path('study-v1-code/worker.jl').read_bytes()).hexdigest()
        assert md['adapters_sha256'] == hashlib.sha256(pathlib.Path('study-v1-code/adapters.jl').read_bytes()).hexdigest()
    else:
        assert md['worker_sha256'] == hashlib.sha256(pathlib.Path('study-v1-code/python_worker.py').read_bytes()).hexdigest()
        assert md['solver_source_sha256'] == hashlib.sha256(pathlib.Path('study-v1-code/python_solver.py').read_bytes()).hexdigest()
    provenance.append(dict(**identity, source_hashes_pass=True, cpu=md['cpu'], input_hashes=md['input_hashes']))
    for label in ['middle', 'path']:
        file = f / (label+'-cpu.txt')
        content = file.read_text() if file.is_file() else ''
        match = re.search(r'raw_stack_words=(\d+)', content)
        words = int(match.group(1)) if match else None
        log = (f/'worker.log').read_text(errors='replace')
        profiles.append(dict(**identity, workload=label, present=bool(content), raw_stack_words=words,
                             buffer_full_warning='profile data buffer is full' in log,
                             allocation_profile_present=(f/'alloc.txt').is_file()))
    memory.append(dict(**identity, **{k: md.get(k) for k in ['warmed_current_rss','warmed_peak_rss','timing_current_rss','timing_peak_rss','profile_peak_rss','whole_worker_peak_rss']},
                       external_whole_worker_peak_rss=p['resource']['peak_rss_bytes']))
    # Compare the actual saved public and cached coefficient byte slices.
    # Both modes have already passed independent original-design certificates.
    fits = (json.loads((f/'fits.json').read_text()) if py else tomllib.loads((f/'fits.toml').read_text()))['records']
    dims = json.loads((pathlib.Path('inputs/pilot-v1')/case/'meta.json' if args.canary else pathlib.Path('inputs/study-v1')/case/('rep'+str(rep))/'meta.json').read_text())
    hashes = {}
    with (f/'fits.bin').open('rb') as stream:
        for r in fits:
            stream.seek(r['offset_bytes'])
            data = stream.read(dims['p']*dims['q']*8)
            assert len(data) == dims['p']*dims['q']*8
            hashes[r['mode'], r['kind'], r['fraction']] = hashlib.sha256(data).hexdigest()
    cs = [r for r in s['certificates'] if r['case']==case and r['rep']==rep and r['method']==method]
    certlookup = {(r['mode'],r['kind'],r['fraction']):r for r in cs}
    for (mode,kind,frac), h in hashes.items():
        if mode != 'public':
            continue
        other = hashes.get(('cached',kind,frac))
        a, b = certlookup['public',kind,frac], certlookup.get(('cached',kind,frac))
        mode_checks.append(dict(**identity,kind=kind,fraction=frac,matched_cached_point=other is not None,
                                bitwise_equal_B=h==other, both_certificates_pass=bool(b and a['passed'] and b['passed']),
                                relative_objective_delta=None if b is None else abs(a['objective']-b['objective'])/max(1.,abs(a['objective']),abs(b['objective']))))
    for frac in [.5,.1,.02]:
        a = lookup.get((case,rep,method,'public','cold',frac))
        b = lookup.get((case,rep,method,'cached','cold',frac))
        if a and b and a['allocated'] is not None and b['allocated'] is not None:
            allocation_witnesses.append(dict(**identity,fraction=frac,public_allocated=a['allocated'],cached_allocated=b['allocated'],
                                             public_minus_cached=a['allocated']-b['allocated']))
    if (f/'counts.toml').is_file():
        if method in ['baseline','candidate']:
            assert md['counter_original_sha256']==md['extension_sha256']
        for r in tomllib.loads((f/'counts.toml').read_text())['records']:
            assert r['matched_timed_B_and_iterations']
            # Preserve either aggregate or per-point counter layouts explicitly.
            raw = r['counts']
            rows = raw if isinstance(raw,list) else [raw]
            total = {}
            for z in rows:
                for k,v in z.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):
                        total[k] = total.get(k,0)+v
            counts.append(dict(**identity,workload=r['workload'],counter_records=len(rows),matched_timed_B_and_iterations=True,**total))
    if (f/'components.csv').is_file():
        cr = list(csv.DictReader((f/'components.csv').open()))
        for name in sorted({r['component'] for r in cr}):
            rs = [r for r in cr if r['component']==name]
            assert len(rs)==3
            components.append(dict(**identity,component=name,seconds=statistics.median(float(r['seconds']) for r in rs),allocated=statistics.median(int(r['bytes']) for r in rs)))

failed_certs = [r for r in s['certificates'] if not r['passed']]
incomplete_profiles = [r for r in profiles if not r['present'] or r['raw_stack_words']==0 or r['buffer_full_warning']]
bad_modes = [r for r in mode_checks if not r['matched_cached_point'] or not r['both_certificates_pass'] or r['relative_objective_delta']>1e-6]
report = dict(expected_processes=s['expected_processes'],successful_processes=s['successful_processes'],failures=s['failures'],
              timing_rows=sum(len(p['timings']) for p in s['processes']),certificates=len(s['certificates']),failed_certificates=failed_certs,
              comparisons=groups,paired_inputs=pairs,provenance=provenance,profiles=profiles,memory=memory,mode_checks=mode_checks,
              allocation_witnesses=allocation_witnesses,counts=counts,components=components,
              incomplete_profiles=incomplete_profiles,bad_mode_checks=bad_modes,
              excluded_incomplete_process_workload_medians=len(s['medians'])-len(medians))
(out/'final-summary.json').write_text(json.dumps(report,indent=2))
for name, rows in [('comparison-summary',groups),('paired-inputs',pairs),('provenance',provenance),('profiles',profiles),('memory',memory),
                   ('mode-checks',mode_checks),('allocation-witnesses',allocation_witnesses),('counts',counts),('components',components)]:
    write_csv(name,rows)
lines = ['# Replicated nuclear PANOC measurements','',
         f"Complete processes: {s['successful_processes']}/{s['expected_processes']}; certificates: {len(s['certificates'])}; failed certificates: {len(failed_certs)}.",
         '','Each independent input contributes one median of three warmed timing rounds. Ratios below one favor the numerator. The 95% interval uses Student t on paired log ratios and estimates a geometric mean ratio; rounds are not additional independent inputs. Intervals are descriptive and not adjusted for multiple comparisons.',
         '', 'All candidates and complete-result comparators include returned diagnostics. The coefficient-only control omits those fields and is attribution evidence only. Public and common cached workloads are reported separately. Public native paths include the same minimal complete-solution extraction adapter before and after.',
         '', '| Case | Comparison | Workload | Inputs | Numerator s | Denominator s | Median ratio | Paired range | Geomean 95% interval |',
         '|---|---|---|---:|---:|---:|---:|---|---|']
def fmt(x):
    return '—' if x is None else f'{x:.5g}'
for r in groups:
    if r['mode']!='public' or (r['kind']=='cold' and r['fraction']!=.1):
        continue
    lines.append('| '+' | '.join([r['case'],r['numerator']+'/'+r['denominator'],r['kind'],str(r['inputs']),fmt(r['numerator_seconds']),fmt(r['denominator_seconds']),fmt(r['median_ratio']),fmt(r['min_ratio'])+'–'+fmt(r['max_ratio']),fmt(r['geomean_95_low'])+'–'+fmt(r['geomean_95_high'])])+' |')
lines += ['', 'Full cold fractions, cached workloads, per-input ratios, adverse results, memory by stage, counters and profiles are in the adjacent CSV/JSON records. Cumulative allocated bytes and whole-process peak RSS measure different quantities; neither substitutes for timing-stage memory.',
          '', f"Profiles absent, empty or buffer-truncated: {len(incomplete_profiles)}. Public/cached point checks with missing or failing certificates/objective mismatch: {len(bad_modes)}. Bitwise equal public/cached coefficients: {sum(r['bitwise_equal_B'] for r in mode_checks)}/{len(mode_checks)}.",
          '', 'The .99/L step, certificate cadence 1/20, quadratic trait and coefficient-only variants are controlled experiments. Their performance does not itself justify changing production defaults. APG remains the package default; production PANOC uses .95/L and cadence 10. The near-fit cancellation guard and Float32 arithmetic are covered by the separate release/LTS correctness study.',
          '', 'This script reads saved outputs only. It performs no generation, fitting or timing reruns, and does not overwrite failed primary receipts. Final adoption/default decisions require interpretation of all cold/path cases and failures.']
(out/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(successful_processes=s['successful_processes'],expected_processes=s['expected_processes'],
                      timing_rows=report['timing_rows'],certificates=report['certificates'],failed_certificates=len(failed_certs),
                      incomplete_profiles=len(incomplete_profiles),bad_mode_checks=len(bad_modes),comparisons=len(groups)),indent=2))
assert not failed_certs and not bad_modes
