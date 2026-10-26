"""Paired process-cluster runtime summaries and evidence gates; Slurm only."""
import csv,json,os,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import tomllib

root=Path(os.environ['HDM_KNOCKOFF_RUNTIME']);reps=[int(x) for x in sys.argv[1].split(',')]
case_names={1:'iid n600 p200',2:'AR(.4) n600 p200',3:'AR(.8) n400 p400',4:'wide AR(.4) n200 p600',5:'fixed AR(.4) n600 p200'}
rows=[];checks=[];metadata=[];resources=[];profiles=[];rrows=[];rchecks=[];roots=[];source_hashes={}
for rep in reps:
    for case in range(1,6):
        for variant in ('baseline','candidate'):
            out=root/f'{variant}-case-{case}-rep-{rep}'
            data=list(csv.DictReader((out/'timings.csv').open()));assert data
            for r in data:
                for k in ('case','rep','input','seed','round','bytes'):r[k]=int(r[k])
                for k in ('seconds','gctime'):r[k]=float(r[k])
                assert r['seconds']>0 and np.isfinite(r['seconds'])
            rows+=data
            c=list(csv.DictReader((out/'checks.csv').open()));assert all(r['valid']=='true' for r in c)
            for r in c:r.update(variant=variant,case=case,rep=rep)
            checks+=c
            profiles_per_worker=len(data)//3;cpu=list(out.glob('profile-*-cpu.txt'));alloc=list(out.glob('profile-*-alloc.txt'))
            assert len(cpu)==len(alloc)==profiles_per_worker
            for paths,kind in ((cpu,'cpu'),(alloc,'allocation')):
                for path in paths:
                    first=path.open().readline().strip();number=int(first.split('=')[1]);assert number>0
                    profiles.append(dict(variant=variant,case=case,rep=rep,kind=kind,path=str(path.relative_to(root)),samples=number))
            witnesses=[r for r in c if r['round']=='0'];assert len(witnesses)==profiles_per_worker
            assert all(int(r['profile_allocation_witness'])>0 for r in witnesses)
            meta=tomllib.loads((out/'metadata.toml').read_text());assert meta['julia_threads']==meta['blas_threads']==1
            metadata.append(meta)
            resource=json.loads((root/f'resource-{variant}-{case}-{rep}.json').read_text());assert resource['returncode']==0
            resources.append(dict(variant=variant,case=case,rep=rep,**resource))
        out=root/f'R-case-{case}-rep-{rep}'
        rd=list(csv.DictReader((out/'timings.csv').open()));assert len(rd)==9
        for r in rd:
            for k in ('case','rep','input','seed','round'):r[k]=int(r[k])
            r['seconds']=float(r['seconds']);assert r['seconds']>0
        rrows+=rd
        rc=list(csv.DictReader((out/'checks.csv').open()));assert len(rc)==12 and all(r['valid'] in ('1','TRUE') for r in rc)
        for r in rc:r.update(case=case,rep=rep)
        rchecks+=rc
        if case<5:
            ri=next(csv.DictReader((out/'root-identity.csv').open()));roots.append(dict(case=case,rep=rep,**ri))
            assert float(ri['root_error'])<1e-8 and float(ri['diagonal_error'])<1e-8 and float(ri['joint_min'])>=-1e-8
        resource=json.loads((root/f'resource-R-{case}-{rep}.json').read_text());assert resource['returncode']==0
        resources.append(dict(variant='R',case=case,rep=rep,**resource))
        for path in out.glob('*-source.txt'):source_hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert (root/f'legacy-check-{rep}.done').read_text().startswith('45 passed')

assert len(rows)==972*len(reps),len(rows)
assert len(profiles)==648*len(reps),len(profiles)
assert len(rrows)==45*len(reps)
assert len(checks)==1296*len(reps)
assert len(metadata)==10*len(reps) and len(resources)==15*len(reps)
assert len({r['seed'] for r in rows})==15*len(reps)

# Three rounds are repeated measurements, not independent Monte Carlo units.
points=defaultdict(list);bytepoints=defaultdict(list)
for r in rows:
    key=(r['variant'],r['case'],r['strategy'],r['payload'],r['rep'],r['input'])
    points[key].append(r['seconds']);bytepoints[key].append(r['bytes'])
for r in rrows:
    key=('R',r['case'],'equicorrelated',r['payload'],r['rep'],r['input'])
    points[key].append(r['seconds'])
assert all(len(v)==3 for v in points.values())
medians={k:float(np.median(v)) for k,v in points.items()}
bytes_med={k:float(np.median(v)) for k,v in bytepoints.items()}
keys=sorted({k[:4] for k in points})
rng=np.random.default_rng(106999000)
bootstrap=rng.integers(0,len(reps),size=(10000,len(reps)))
absolute=[]
for variant,case,strategy,payload in keys:
    process=[np.median([medians[(variant,case,strategy,payload,rep,i)] for i in (1,2,3)]) for rep in reps]
    b=[np.median([bytes_med[(variant,case,strategy,payload,rep,i)] for i in (1,2,3)]) for rep in reps] if variant!='R' else [np.nan]
    absolute.append(dict(variant=variant,case=case,name=case_names[case],strategy=strategy,payload=payload,
                         median_seconds=float(np.median(process)),min_process_seconds=float(min(process)),max_process_seconds=float(max(process)),
                         median_heap_bytes=float(np.median(b))))
comparisons=[];rep_ratios=[]

def compare(case,kind,numerator,reference):
    # numerator/reference each specify (variant,strategy,payload).
    values=[]
    for rep in reps:
        ratios=[medians[(numerator[0],case,numerator[1],numerator[2],rep,i)]/
                medians[(reference[0],case,reference[1],reference[2],rep,i)] for i in (1,2,3)]
        ratio=float(np.median(ratios));values.append(ratio)
        rep_ratios.append(dict(case=case,comparison=kind,rep=rep,ratio=ratio))
    v=np.asarray(values);lo,hi=np.quantile(np.median(v[bootstrap],axis=1),[.025,.975])
    comparisons.append(dict(case=case,name=case_names[case],comparison=kind,ratio=float(np.median(v)),lo95=float(lo),hi95=float(hi),
                            min_process_ratio=float(min(v)),max_process_ratio=float(max(v)),independent_process_pairs=len(reps)))

for case in range(1,6):
    payloads=sorted({k[3] for k in keys if k[:3]==('baseline',case,'equicorrelated')})
    for payload in payloads:
        compare(case,f'candidate/baseline:{payload}',('candidate','equicorrelated',payload),('baseline','equicorrelated',payload))
    for strategy in ('mvr','maxent'):
        for payload in ('diagonal','sampling','construct','filter')+(() if case==5 else ('construct-rng',)):
            compare(case,f'{strategy}/equicorrelated:{payload}',('candidate',strategy,payload),('candidate','equicorrelated',payload))
        compare(case,f'{strategy} validated-factory/raw-optimizer',('candidate',strategy,'diagonal'),('candidate',strategy,'raw-optimizer'))
    payload='construct-unitnorm' if case==5 else 'construct-rng'
    compare(case,f'R/candidate:{payload}',('R','equicorrelated',payload),('candidate','equicorrelated',payload))

def write_csv(name,data):
    with (root/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

write_csv('absolute-times.csv',absolute);write_csv('paired-ratios.csv',comparisons);write_csv('replication-ratios.csv',rep_ratios)
write_csv('profile-index.csv',profiles);write_csv('resource-summary.csv',resources);write_csv('R-root-identities.csv',roots)
filters=[r for r in checks if r['payload']=='filter' and r['round']!='0']
raw=[r for r in checks if r['payload']=='raw-optimizer' and r['round']!='0']
summary=dict(shared_fresh_inputs=15*len(reps),independent_process_replications=len(reps),native_workers=10*len(reps),R_workers=5*len(reps),
             native_timing_rows=len(rows),R_timing_rows=len(rrows),native_check_rows=len(checks),R_check_rows=len(rchecks),
             complete_legacy_field_input_matrix_assertions=45*len(reps),CPU_profiles=len(profiles)//2,allocation_profiles=len(profiles)//2,
             allocation_witnesses=len(profiles)//2,actual_Julia_metadata=len(metadata),successful_resource_receipts=len(resources),
             filter_certificates=len(filters),max_filter_residual=max(float(r['residual']) for r in filters),
             raw_optimizer_fulljoint_certificates=len(raw),max_raw_fulljoint_residual=max(float(r['residual']) for r in raw),
             R_root_identities=len(roots),max_R_root_error=max(float(r['root_error']) for r in roots),source_hashes=source_hashes)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Knockoff construction runtime evidence','',
 f'{len(reps)} independent process replications per case/backend, three fresh shared inputs per replication, three warmed rounds per input. The {15*len(reps)} distinct input seeds are shared across procedures. Ratios first take input medians over rounds, then paired medians over three inputs within each process replication, then the median of seven process ratios. Intervals resample the seven process clusters 10,000 times; individual rounds are not independent observations. Actual source/CPU/Julia/thread records and every timing round remain available.','',
 '## Existing equicorrelated API','',
 '| Case | Payload | Candidate / baseline | 95% process interval | Process range |','|---|---|---:|---|---|']
for r in comparisons:
    if not r['comparison'].startswith('candidate/baseline:'):continue
    lines.append(f"| {r['name']} | {r['comparison'].split(':')[1]} | {r['ratio']:.3f} | [{r['lo95']:.3f}, {r['hi95']:.3f}] | {r['min_process_ratio']:.3f}–{r['max_process_ratio']:.3f} |")
lines+=['','Ratios above one are regressions and remain in the evidence. Low-level constructors retain the same numerical arithmetic but now obtain owned construction metadata internally; full filters expose that metadata. This feature adds choices and validation, not a new faster optimizer. Complete legacy result fields, constructor matrices and input bytes are equal in all315 assertions. The independent fulljoint checks and profiling are outside timers.','',
 '## Cost of choosing another strategy','',
 '| Case | Ratio | Median | 95% process interval |','|---|---|---:|---|']
for r in comparisons:
    if not r['comparison'].endswith((':construct',':filter',':diagonal')) or r['comparison'].startswith(('candidate','R/')):continue
    lines.append(f"| {r['name']} | {r['comparison']} | {r['ratio']:.3f} | [{r['lo95']:.3f}, {r['hi95']:.3f}] |")
lines+=['','Sampling is the public supplied-diagonal constructor, including its required feasibility/factorization work; it is not a cached matrix multiply. Raw optimizer measurements omit the adapter certificate/factory work; their outputs are independently certified outside timers. The validated-factory/raw-optimizer ratios in paired-ratios.csv measure that additional work and do not claim an algorithmic speed advantage. No covariance cache or Cholesky fast path is introduced.','',
 '## Applicable public R constructors','',
 '| Case | R / candidate | 95% process interval |','|---|---:|---|']
for r in comparisons:
    if not r['comparison'].startswith('R/'):continue
    lines.append(f"| {r['name']} | {r['ratio']:.3f} | [{r['lo95']:.3f}, {r['hi95']:.3f}] |")
lines+=['','R Gaussian and native construct-rng both include RNG setup/draw; language RNG algorithms differ. Both receive identical X/covariance inputs. R fixed and native construct-unitnorm receive identical binary unit-norm designs. R fixed uses the unshrunk SVD boundary policy, native safety1e-8; R Gaussian default regularization also differs slightly. Their actual root/joint/Gram identities are checked, and installed R source is retained. No R MVR/maxent constructor with the same objective is available in this package; their independent objective comparison is the conic oracle study. Matched R glmnet fits use the separate exact saved augmented matrices from the statistical cohort.','',
 '## Correctness, profiles and memory','',
 f'All {len(filters)} runtime full-filter certificates pass1e-7 (max{summary["max_filter_residual"]:.6g}); all {len(raw)} timed raw optimizer certificates pass1e-6 (max{summary["max_raw_fulljoint_residual"]:.6g}). All {summary["R_root_identities"]} R root/joint witnesses pass (max root error{summary["max_R_root_error"]:.6g}), and every timed R fixed result passes both Gram identities with its actual equicorrelated diagonal. There are {summary["CPU_profiles"]} nonempty CPU and {summary["allocation_profiles"]} nonempty heap profile reports, separate from timing, each with a positive direct allocation witness. Profiles are observational and do not turn inclusive stack counts into causal time percentages.','',
 '| Case | Backend | Median maximum RSS MiB | Range MiB |','|---|---|---:|---|']
for case in range(1,6):
    for variant in ('baseline','candidate','R'):
        v=[r['max_rss_bytes']/2**20 for r in resources if r['case']==case and r['variant']==variant]
        lines.append(f'| {case_names[case]} | {variant} | {np.median(v):.1f} | {min(v):.1f}–{max(v):.1f} |')
lines+=['','Maximum RSS includes imports/JIT,warmup,timers,certificates and separate profiles, and records the largest worker/descendant resident size, not a sum of simultaneous child processes or fitted-state memory. Julia baseline and candidate both import the same pinned Knockoffs dependency for this controlled study; R has a different runtime/import/observer footprint. Heap bytes cover Julia-managed allocation and exclude foreign allocations. No cross-language fitted-state memory advantage is claimed.','']
(root/'TABLES.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'}))
