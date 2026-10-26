"""Aggregate saved measurements only; no new fit, profile, certificate or time."""
import csv,gzip,json,pathlib,statistics,tarfile,tomllib
import numpy as np
root=pathlib.Path('results/study-report-v1');out=pathlib.Path('results/completion-v1');out.mkdir(exist_ok=True)
s=json.loads((root/'final-summary.json').read_text())
assert s['successful_processes']==s['expected_processes']==1800
assert not s['failures'] and not s['failed_certificates'] and not s['incomplete_profiles'] and not s['bad_mode_checks']
def med(rows,key):
    x=[r[key] for r in rows if r.get(key) is not None]
    return statistics.median(x) if x else None
def write(name,rows):
    if not rows:return
    with (out/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
memory=[];components=[];counters=[];allocations=[]
for case in sorted({r['case'] for r in s['memory']}):
    for method in sorted({r['method'] for r in s['memory']}):
        rs=[r for r in s['memory'] if r['case']==case and r['method']==method]
        memory.append(dict(case=case,method=method,inputs=len(rs),**{k:med(rs,k) for k in ['warmed_current_rss','warmed_peak_rss','timing_current_rss','timing_peak_rss','profile_peak_rss','whole_worker_peak_rss','external_whole_worker_peak_rss']}))
        cr=[r for r in s['components'] if r['case']==case and r['method']==method]
        for name in sorted({r['component'] for r in cr}):
            rs=[r for r in cr if r['component']==name]
            components.append(dict(case=case,method=method,component=name,inputs=len(rs),seconds=med(rs,'seconds'),allocated=med(rs,'allocated')))
        for workload in ['public/cold/0.1','public/path/0.0']:
            rs=[r for r in s['counts'] if r['case']==case and r['method']==method and r['workload']==workload]
            if rs:
                keys=sorted({k for r in rs for k in r if k not in ['case','rep','method','workload','matched_timed_B_and_iterations']})
                counters.append(dict(case=case,method=method,workload=workload,inputs=len(rs),**{k:med(rs,k) for k in keys}))
    for mode in ['public','cached']:
        for kind,frac in [('cold',.5),('cold',.1),('cold',.02),('path',0.)]:
            rs=[r for r in s['paired_inputs'] if r['case']==case and r['numerator']=='candidate' and r['denominator']=='baseline' and r['mode']==mode and r['kind']==kind and r['fraction']==frac]
            allocations.append(dict(case=case,mode=mode,kind=kind,fraction=frac,inputs=len(rs),candidate_allocated=med(rs,'numerator_allocated'),baseline_allocated=med(rs,'denominator_allocated')))
certs=list(csv.DictReader(pathlib.Path('results/study-v1/certificates.csv').open()))
certlookup={(r['case'],int(r['rep']),r['method'],r['mode'],r['kind'],float(r['fraction'])):r for r in certs}
pointpairs=[]
for case in sorted({r['case'] for r in s['memory']}):
    for rep in range(1,21):
        parent=pathlib.Path('results/study-v1')/case/('rep'+str(rep))
        before=tomllib.loads((parent/'baseline/fits.toml').read_text())['records']
        after=tomllib.loads((parent/'candidate/fits.toml').read_text())['records']
        lookup={(r['mode'],r['kind'],r['fraction']):r for r in after}
        dims=json.loads((pathlib.Path('inputs/study-v1')/case/('rep'+str(rep))/'meta.json').read_text())
        count=dims['p']*dims['q']
        with (parent/'baseline/fits.bin').open('rb') as bf,(parent/'candidate/fits.bin').open('rb') as af:
            for b in before:
                key=(b['mode'],b['kind'],b['fraction']);a=lookup[key]
                bf.seek(b['offset_bytes']);af.seek(a['offset_bytes'])
                B=np.frombuffer(bf.read(count*8),dtype='<f8');A=np.frombuffer(af.read(count*8),dtype='<f8')
                b0=np.frombuffer(bf.read(dims['q']*8),dtype='<f8');a0=np.frombuffer(af.read(dims['q']*8),dtype='<f8')
                bc=certlookup[(case,rep,'baseline',*key)];ac=certlookup[(case,rep,'candidate',*key)]
                bo,ao=float(bc['objective']),float(ac['objective'])
                pointpairs.append(dict(case=case,rep=rep,mode=key[0],kind=key[1],fraction=key[2],
                    relative_B_difference=float(np.linalg.norm(A-B)/max(1.,np.linalg.norm(B))),
                    max_intercept_difference=float(np.max(np.abs(a0-b0))),
                    relative_objective_difference=abs(ao-bo)/max(1.,abs(ao),abs(bo)),
                    equal_rank=a['rank']==b['rank'],equal_df=a['df']==b['df'],bitwise_equal_B=np.array_equal(A,B)))
write('before-after-points',pointpairs)
targets=[r for r in s['comparisons'] if r['numerator']=='candidate' and r['denominator']=='baseline']
apg=[r for r in s['comparisons'] if r['numerator']=='candidate' and r['denominator']=='apg' and r['mode']=='public']
witness=[r for r in s['allocation_witnesses'] if r['method'] in ['baseline','candidate']]
report=dict(successful_processes=1800,timing_rows=s['timing_rows'],certificates=len(certs),
    max_relative_gap=max(float(r['relative_gap']) for r in certs),max_normalized_mapping=max(float(r['normalized_mapping']) for r in certs),
    mode_checks=len(s['mode_checks']),bitwise_equal_mode_checks=sum(r['bitwise_equal_B'] for r in s['mode_checks']),
    profiles=len(s['profiles']),incomplete_profiles=s['incomplete_profiles'],provenance_records=len(s['provenance']),
    native_setup_witnesses=len(witness),nonpositive_native_setup_allocation_differences=[r for r in witness if r['public_minus_cached']<=0],
    target_groups=len(targets),target_groups_with_95_upper_at_least_one=[r for r in targets if r['geomean_95_high']>=1],
    target_inputs_slower=[r for r in s['paired_inputs'] if r['numerator']=='candidate' and r['denominator']=='baseline' and r['ratio']>1],
    public_apg_comparisons=apg,memory=memory,components=components,counters=counters,allocations=allocations,
    before_after_points=len(pointpairs),max_before_after_relative_B_difference=max(r['relative_B_difference'] for r in pointpairs),
    max_before_after_relative_objective_difference=max(r['relative_objective_difference'] for r in pointpairs),
    before_after_rank_or_df_differences=[r for r in pointpairs if not r['equal_rank'] or not r['equal_df']])
(out/'summary.json').write_text(json.dumps(report,indent=2))
for name,rows in [('memory-summary',memory),('component-summary',components),('counter-summary',counters),('allocations',allocations),('public-apg',apg)]:write(name,rows)
print(json.dumps({k:v for k,v in report.items() if k not in ['memory','components','counters','allocations','public_apg_comparisons','target_inputs_slower']},indent=2))
print('TARGET_SLOWER_OBSERVATIONS',len(report['target_inputs_slower']))
# Compact tables/manifests/profiles, retaining every original outcome. Raw
# coefficient/intercept binaries and frozen input bytes remain remotely saved.
paths=list(root.rglob('*'))+list(out.glob('*.csv'))+list(out.glob('*.json'))
paths += list(pathlib.Path('results/study-v1').glob('*.csv'))+list(pathlib.Path('results/study-v1').glob('*.tsv'))
for f in pathlib.Path('results/study-v1').glob('*/rep*/*/*'):
    if f.is_file() and f.name!='fits.bin':paths.append(f)
with tarfile.open('completion-report-v1.tar.gz','w:gz') as a:
    for f in sorted(set(paths)):
        if f.is_file():a.add(f,arcname=str(f),recursive=False)
