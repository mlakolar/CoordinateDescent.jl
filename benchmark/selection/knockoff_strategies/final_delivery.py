"""Save reviewable reports and compact reporting inputs; no new fitting."""
import csv,json,os,hashlib,shutil,tarfile
from pathlib import Path
import numpy as np
import tomllib
base=Path.cwd();out=base/'final-delivery-v1';assert not out.exists();out.mkdir()
stat=base/'stat-main-v1';runtime=base/'runtime-main-v1';order=base/'runtime-order-check-v1'
s=json.loads((stat/'summary.json').read_text());t=json.loads((runtime/'summary.json').read_text());o=json.loads((order/'summary.json').read_text())
obj=json.loads((base/'objective-suite-v1'/'assessment.json').read_text());sampler=json.loads((base/'sampler-reference-v2'/'summary.json').read_text())
assert s['inputs']==16128 and s['valid_certificates']==48384 and s['R_valid']==1008 and not s['R_selection_disagreements']
assert t['shared_fresh_inputs']==105 and t['CPU_profiles']==t['allocation_profiles']==2268
assert o['inputs_reused']==105 and o['extra_independent_datasets']==0 and o['fresh_processes']==70
assert o['legacy_assertions']==315

for source,name in [(stat/'TABLES.md','STATISTICS.md'),(runtime/'TABLES.md','RUNTIME.md'),(order/'TABLES.md','ORDER-SENSITIVITY.md'),
 (base/'objective-suite-v1'/'ASSESSMENT.md','OBJECTIVES.md')]:
    shutil.copy2(source,out/name)
text=(out/'RUNTIME.md').read_text().replace('Ratios above one are regressions and remain in the evidence.','Ratios above one remain in the evidence. The diagonal row compares the NEW validated factory with the unchanged legacy vector selector, rather than measuring a regression to that selector. The primary baseline/candidate launch had a variant-phase confound; consult ORDER-SENSITIVITY.md before interpreting small before/after speed differences.')
(out/'RUNTIME.md').write_text(text)
for source in ['monte-carlo.csv','paired-differences.csv','worker-means.csv']:shutil.copy2(stat/source,out/source)
for source in ['absolute-times.csv','paired-ratios.csv','replication-ratios.csv','resource-summary.csv','R-root-identities.csv']:shutil.copy2(runtime/source,out/f'runtime-{source}')
shutil.copy2(order/'paired-ratios.csv',out/'order-paired-ratios.csv')
shutil.copy2(base/'sampler-reference-v2'/'sampler-certificates.csv',out/'sampler-certificates.csv')
shutil.copy2(base/'objective-suite-v1'/'independent-objectives.csv',out/'independent-objectives.csv')
mc=list(csv.DictReader((stat/'monte-carlo.csv').open()));paired=list(csv.DictReader((stat/'paired-differences.csv').open()))

def point(name,strategy):return next(r for r in mc if r['name']==name and r['strategy']==strategy)
def difference(name,strategy,metric='power'):return next(r for r in paired if r['name']==name and r['method']==strategy and r['reference']=='equicorrelated' and r['metric']==metric)

# Supplement empirical zero-SE rows with conservative rare-event information.
# These are reporting bounds, not another simulation or outcome-based tuning.
zero=[]
for sc in range(1,17):
    groups={m:[] for m in ('equicorrelated','mvr','maxent')}
    for shard in range(1,8):
        for r in csv.DictReader((stat/f'scenario-{sc}-shard-{shard}'/'certificates.csv').open()):groups[r['strategy']].append(r)
    for method,rows in groups.items():
        for metric in ('fdp','power'):
            x=np.asarray([float(r[metric]) for r in rows]);x=x[np.isfinite(x)]
            if len(x) and np.all(x==0):
                upper=1.96**2/(len(x)+1.96**2)
                zero.append(dict(scenario=sc,method=method,metric=metric,independent_inputs=len(x),observed_nonzero=0,
                  empirical_mean=0.,empirical_se=0.,wilson_nonzero_probability_upper95=upper,conservative_mean_upper95=upper))
with (out/'zero-event-bounds.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(zero[0]));w.writeheader();w.writerows(zero)

p=point('ar04-alt','mvr');d=difference('ar04-alt','mvr');a=point('ar04-alt','equicorrelated')
lines=['# MVR and maximum-entropy knockoff construction: Discovery evidence, 2026-10-26','',
 'Explicit optimized diagonal choices are useful, but the evidence supports retaining equicorrelation as the default. The optional Knockoffs.jl3.1.4 adapter supplies only diagonal optimization. Sampling, feasibility checks, joint lasso and knockoff+ remain native. The change adds an owned diagonal/diagnostics API, forwards construction options through the filter, and rejects failed original-objective stationarity without silently changing strategy. It does not implement a new diagonal optimizer or a covariance cache.','',
 '## Statistical benefit and adverse results','',
 f'The main cohort contains16,128 independent Gaussian datasets:1,008 per scenario across seven independent process shards. Only the diagonal changes; X,y,independent normal innovations,lambda=.04,cyclic lasso controls and knockoff+q=.1 are shared. All48,384 native original-design certificates pass (maximum residual{s["max_residual"]:.6g}); there are no internal stopping flags, fit errors or failed public-filter forwarding witnesses. R glmnet5.1 receives the exact saved augmented matrices on336 of those inputs: all1,008 comparator fits pass, all selected sets agree, maximum relative objective difference{s["R_max_relative_objective_difference"]:.6g}. The R subset is not additional independent data.','',
 f'For AR(.4) with alternating signals, MVR power rises from{float(a["power_mean"]):.4f} to{float(p["power_mean"]):.4f}: paired gain{float(d["difference"]):.4f},95% Monte Carlo interval[{float(d["lo95"]):.4f},{float(d["hi95"]):.4f}]. MVR estimated FDR is{float(p["fdp_mean"]):.4f}(MC SE{float(p["fdp_se"]):.4f}). This supports exposing an explicit alternative to the old filter policy.','',
 '| Design | Equicorrelated power | MVR power | Maximum-entropy power | MVR minus equicorrelation,95% MC interval |',
 '|---|---:|---:|---:|---|']
for name in ('iid-alt','ar04-alt','ar04-same','ar08-alt','ar08-same','weak-ar04-alt','weak-ar04-same','wide-ar04-alt','wide-ar04-same'):
    q=[point(name,m) for m in ('equicorrelated','mvr','maxent')];z=difference(name,'mvr')
    lines.append(f'| {name} | {float(q[0]["power_mean"]):.4f} | {float(q[1]["power_mean"]):.4f} | {float(q[2]["power_mean"]):.4f} | [{float(z["lo95"]):.4f},{float(z["hi95"]):.4f}] |')
lines+=['','Same-sign power losses are substantial and remain in the recommendation. Severe-correlation and wide alternating-signal power remain poor. Three explicit covariance-misspecification stress cases are reported separately; they establish no misspecified-model FDR guarantee. Global-null power is undefined, and null FDR intervals use Wilson binomial intervals. All-sample empirical zero SEs do not prove an event impossible; zero-event-bounds.csv supplies conservative rare-event bounds. Full FDR/power/selection counts, paired differences and worker means are in STATISTICS.md and the CSVs. No automatic strategy selection or statistical guarantee beyond the stated construction/statistic assumptions is introduced.','',
 '## Construction correctness and optimizer controls','',
 'Julia1.10.12 and1.13.2 each pass556 core/selection/golden/construction checks plus587 optional-extension checks. They cover Float32/64,unequal scales,1e-20/1/1e20 units,supplied/zero/boundary diagonals,conflicts,owned inputs,seeded RNG,supplied noise,fixed Gram identities and explicit optimizer failure. The Float32 legacy promotion and all existing result fields/old positional constructors are preserved.','',
 'Sixteen fixed covariance matrices compare raw and adapted optimizers under package-default,proposed,robust and tighter controls. All64 proposed/robust adapted results are feasible; maximum independent fulljoint gradient ratio is4.812e-9. All20 small CVXPY/CLARABEL problems report optimal and are feasible; relative objective differences are at most6.24e-10. These are numerical comparators,not exact global error bounds. All raw package-default results are feasible,but30/32 fail the adapter original-objective stationarity requirement (maximum raw fulljoint ratio4.744e-5). Their failed outputs/errors remain in OBJECTIVES.md and the reporting bundle.','',
 'The independent conditional-map study passes16/16 native and16/16 R samplers (maximum root errors4.020e-12 and1.702e-12). The pinned Knockoffs.jl full sampler passes only1/16; its orientation defect remains,with maximum relative root error.999624 on this unequal-scale suite. It is excluded from valid FDR rankings. Its diagonal optimizer remains usable with the independently validated native sampler. R Gaussian defaults differ slightly from native (maximum measured diagonal difference1.012e-6); R fixed uses an unshrunk SVD boundary diagonal while native retains safety1e-8. Actual maps,Gram identities and installed source hashes are retained.','',
 '## Runtime and memory','',
 f'The primary runtime cohort has105 fresh shared inputs,seven process replications,three inputs per process and three warmed rounds:70 native and35 R workers,6,804 native and315 R timing rows. All1,260 runtime filter certificates pass (max{t["max_filter_residual"]:.6g}); all630 timed raw optimizer fulljoint certificates pass (max{t["max_raw_fulljoint_residual"]:.6g}). There are2,268 nonempty CPU and2,268 nonempty allocation reports,2,268 positive direct allocation witnesses,70 actual Julia/CPU/thread metadata records and105 successful resource receipts.','',
 'The primary launch placed baseline workers before candidate workers. All primary observations are preserved. A separately declared alternating-order sensitivity reuses the same105 inputs in70 fresh processes,without pooling them as another data cohort. Each process pair runs both variants serially on the same allocated CPU and alternates which runs first. Both studies preserve all315 complete legacy input/matrix/result-field assertions. Use ORDER-SENSITIVITY.md when assessing small before/after differences; method/payload order within candidate workers was already rotated for the primary strategy comparisons.','',
 '| Runtime case | MVR / equicorrelated full filter | Maximum entropy / equicorrelated full filter |','|---|---:|---:|']
ratios=list(csv.DictReader((runtime/'paired-ratios.csv').open()))
for case in range(1,6):
    x=[next(r for r in ratios if int(r['case'])==case and r['comparison']==f'{m}/equicorrelated:filter') for m in ('mvr','maxent')]
    lines.append(f'| {x[0]["name"]} | {float(x[0]["ratio"]):.3f} [{float(x[0]["lo95"]):.3f},{float(x[0]["hi95"]):.3f}] | {float(x[1]["ratio"]):.3f} [{float(x[1]["lo95"]):.3f},{float(x[1]["hi95"]):.3f}] |')
lines+=['','These optimized choices have a real runtime cost in this study. Preparation,sampling,full construction,full filtering and raw-optimizer costs are separated in RUNTIME.md. The diagonal factory row includes feasibility/objective certification and is distinct from the unchanged legacy vector selector. Applicable public R constructors receive identical X/covariance or binary unit-norm fixed designs; their RNG algorithms and stated diagonal defaults differ. R has no matching MVR/maxent optimizer here; the independent convex objective comparator covers those objectives. Heap allocation excludes foreign memory. Worker maximum RSS includes imports/JIT,checks and profiles and is not fitted-state memory; no cross-language memory advantage is claimed.','',
 '## Recommended defaults','',
 '| Setting | Recommendation | Evidence/limit |','|---|---|---|',
 '| strategy | retain equicorrelation | Mixed-sign gains accompany same-sign losses and larger optimized-procedure cost;no universal winner |',
 '| equicorrelation safety | retain1e-8 | Existing seeded arithmetic/precision and scale-relative boundaries remain compatible |',
 '| optimized backend | niter1000,tol1e-10,lambda_min1e-10,robustfalse | Original-objective certificates pass;package-default stopping does not supply that assurance |',
 '| independent optimality threshold | max(1e-6,100eps(input precision)) | Certify actual rounded diagonal/original covariance;Float32 needs its precision margin |',
 '| q / threshold | retain.1 / knockoff+offset1 | Compatibility;this is a target,not an empirical guarantee under arbitrary misspecification |',
 '| lasso penalty / fitting | retain explicitly supplied positive lambda;do not retune CDOptions globally | This study fixes.04 and stricter comparison controls;it supplies no data-driven penalty selector |','',
 'Backend actual iteration count/convergence status are unavailable and remain labeled unavailable. The independent stationarity result is separate. Robust updates are explicit user options,not a silent fallback. Gaussian model-X still requires the supplied population law;fixed-X still requires full rank,n>=2p and the documented sufficiency/statistic/error assumptions. No covariance estimator or factor cache is added.','',
 '## Recreation and retained evidence','',
 'Scientific production source is candidate2 atf14306b8b0350428f2cbb03c8126347fe84f0a3e,with baseline295fbf5420d6384a1ca261023367ee86d0a40b8c. Source snapshots,actual environment manifests,source hashes,all reporting rows,failures and job receipts are in reporting-inputs-v1.tar.gz;all primary worker profiles are in worker-profiles-v1.tar.gz. The same raw X/y/innovations,coefficient and constructor/result-field records remain on Discovery under the recorded run directory. They are not routinely archived or transferred. The source/seed/environment recipe recreates them.','',
 'Failed parse/dependency/stdlib/resource-wrapper canaries,old R1e-14 certificate failures,the R CSV-dimname failure and the incorrect fixed-diagonal helper assumption remain recorded. None caused outcome-based main-cohort signal/sequence/penalty/tolerance tuning. The alternating-order sensitivity is explicitly separate. No primary outputs are replaced and no merge/auto-merge is performed.','']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(out/'summary.json').write_text(json.dumps(dict(statistics=s,runtime=t,order_sensitivity=o,objectives=obj,samplers=sampler),indent=2)+'\n')

selected={}
def include(path,arcname=None):
    if path.is_file():selected[arcname or str(path.relative_to(base))]=path
for path in out.iterdir():include(path,'final-report/'+path.name)
for name in ('summary.json','TABLES.md','monte-carlo.csv','paired-differences.csv','worker-means.csv','R-check-exit-status.txt'):
    include(stat/name)
for work in stat.glob('scenario-*-shard-*'):
    for name in ('certificates.csv','fit-status.csv','construction-status.csv','metadata.toml','R-certificates.csv','R-status.csv','R-session.txt','R-summary.json','check-summary.json'):include(work/name)
for directory in (runtime,order):
    for path in directory.glob('*.csv'):include(path)
    for path in directory.glob('*.json'):include(path)
    for path in directory.glob('*.tsv'):include(path)
    for path in directory.glob('*.done'):include(path)
    include(directory/'TABLES.md')
    for work in directory.glob('*-case-*-rep-*'):
        for pattern in ('timings.csv','checks.csv','metadata.toml','session.txt','root-identity.csv','diagonal-*.csv','*-source.txt'):
            for path in work.glob(pattern):include(path)
for work in (base/'objective-suite-v1').glob('case-*'):
    for pattern in ('*.csv','*.toml'):
        for path in work.glob(pattern):include(path)
for pattern in ('*.csv','*.json','*.md','*.tsv'):
    for path in (base/'objective-suite-v1').glob(pattern):include(path)
for pattern in ('*.csv','*.json','*.toml','*source.txt','*session.txt'):
    for path in (base/'sampler-reference-v2').glob(pattern):include(path)
for directory in base.glob('*results'):
    for pattern in ('*.log','*.toml'):
        for path in directory.glob(pattern):include(path)
for directory in (base/'runtime-pilot-v1',base/'runtime-pilot-v2'):
    for pattern in ('*.json','*.tsv'):
        for path in directory.glob(pattern):include(path)
    for work in directory.glob('*-case-*-rep-*'):
        for pattern in ('timings.csv','checks.csv','metadata.toml','session.txt','root-identity.csv','diagonal-*.csv','*-source.txt'):
            for path in work.glob(pattern):include(path)
for directory in (base/'runtime-pilot-v1',base/'runtime-pilot-v2',base/'stat-pilot-v1'):
    for logdir in directory.glob('*logs'):
        for name in ('stdout','stderr'):
            for path in logdir.rglob(name):include(path)
include(base/'registry-snapshot-before-delivery.json')
for path in base.glob('*.out'):
    if not path.name.startswith('final-delivery'):include(path)
for path in base.glob('*.tar'):include(path,'source-archives/'+path.name)
for env in base.glob('env-*'):
    for name in ('Project.toml','Manifest.toml'):include(env/name)
for work in (base/'stat-pilot-v1').glob('scenario-*-shard-*'):
    for pattern in ('*.csv','*.json','*.toml','*session.txt'):
        for path in work.glob(pattern):include(path)
for name in ('ProximalBase.jl','ProximalOPT.jl','KernelSmoothing.jl'):
    include(base.parent/'issue99'/name/'Project.toml','dependency-source/'+name+'/Project.toml')
for name in ('Project.toml','src/knockoffs.jl','src/lasso.jl'):
    include(base.parent/'issue99'/'baseline'/name,'baseline-source/'+name)
# Select only the actually matched installed backend source, never a whole depot.
expected=tomllib.loads((base/'objective-suite-v1'/'case-1'/'metadata.toml').read_text())['backend_utilities_sha256']
for depot in (base/'depot',base.parent.parent/'20261025-cd82-compare'/'depot',base.parent.parent/'20261025-cd82-selection'/'depot'):
    for package in (depot/'packages'/'Knockoffs').glob('*'):
        utilities=package/'src'/'utilities.jl'
        if utilities.is_file() and hashlib.sha256(utilities.read_bytes()).hexdigest()==expected:
            for name in ('Project.toml','LICENSE','src/utilities.jl','src/modelX.jl'):include(package/name,'backend-source/'+name)
assert 'backend-source/src/utilities.jl' in selected
manifest=[dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in sorted(selected.items())]
with (out/'reporting-file-manifest.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
include(out/'reporting-file-manifest.csv','final-report/reporting-file-manifest.csv')
with tarfile.open(out/'reporting-inputs-v1.tar.gz','w:gz',compresslevel=6) as archive:
    for name,path in sorted(selected.items()):archive.add(path,arcname=name,recursive=False)
profile_paths=sorted(runtime.glob('*-case-*-rep-*/profile-*-cpu.txt'))+sorted(runtime.glob('*-case-*-rep-*/profile-*-alloc.txt'))
assert len(profile_paths)==4536
with tarfile.open(out/'worker-profiles-v1.tar.gz','w:gz',compresslevel=6) as archive:
    for path in profile_paths:archive.add(path,arcname=str(path.relative_to(base)),recursive=False)
receipts=[dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (out/'reporting-inputs-v1.tar.gz',out/'worker-profiles-v1.tar.gz')]
(out/'delivery.json').write_text(json.dumps(dict(reporting_files=len(selected),profiles=len(profile_paths),artifacts=receipts),indent=2)+'\n')
print((out/'delivery.json').read_text())
