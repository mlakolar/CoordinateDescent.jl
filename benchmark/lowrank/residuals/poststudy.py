"""Inspect retained reports only, under Slurm; never refit or retune."""
import csv,json,pathlib,re,statistics,sys,tomllib
root=pathlib.Path(sys.argv[1]);base=root/'results/main'
summary=json.loads((base/'summary.json').read_text())
profiles=[]
for p in sorted(base.glob('*-*/*/cpu-*.txt')):
    text=p.read_text();label,variant=p.parts[-3:-1]
    row=dict(input=label,variant=variant,mode=p.stem[4:])
    for key,pattern in [('calls',r'calls=(\d+)'),('snapshots',r'Total snapshots: (\d+)')]:
        m=re.search(pattern,text);row[key]=int(m.group(1)) if m else None
    for key,name in [('design_response_svd','_lr_svd'),('lapack_gesdd','gesdd!'),('blas_gemm','gemm!')]:
        row[key+'_inclusive_max']=max([int(s.split()[0]) for s in text.splitlines() if name in s and s.split()[0].isdigit()],default=0)
    profiles.append(row)
with (base/'profile-counts.csv').open('w') as f:
    w=csv.DictWriter(f,list(profiles[0]));w.writeheader();w.writerows(profiles)
metadata=[]
for p in sorted(base.glob('*-*/*/metadata.toml')):metadata.append(tomllib.loads(p.read_text()))
expected_sources={'baseline':'b611a82eb5228e1f33ca2d1dce66aa682187764025519e648a759fa2461de448',
                  'candidate':'25de29750698de3dbab963211df663759fe44a48593d09dae9f96df0095dd765'}
assert len(metadata)==240
assert all(m['source_sha256']==expected_sources[m['variant']] and m['julia_threads']==m['blas_threads']==1 for m in metadata)
assert {m['cpu'] for m in metadata}=={'AMD EPYC 7513 32-Core Processor'}
certificates=list(csv.DictReader((base/'certificates.csv').open()))
native=[r for r in certificates if r['variant'] in ('baseline','candidate')]
error_fields=['relative_loss_error','relative_coefficient_error','relative_prediction_error','relative_minimum_norm_error','residual_error','intercept_error','df_error','exact_df_error']
error_max={k:max(float(r[k]) for r in native) for k in error_fields}
allocations=[]
for case in ('responses','wide'):
    for mode in ('public_fit','public_path'):
        z={r['variant']:r for r in summary['table'] if r['case']==case and r['mode']==mode}
        allocations.append(dict(case=case,mode=mode,baseline_bytes=z['baseline']['median_allocated_bytes'],candidate_bytes=z['candidate']['median_allocated_bytes']))
result=dict(native_source_hashes=expected_sources,native_metadata_records=len(metadata),native_error_maxima=error_max,
    julia_profile_snapshots=sum(r['snapshots'] or 0 for r in profiles if r['variant'] in ('baseline','candidate')),
    julia_cpu_profiles=sum(r['variant'] in ('baseline','candidate') for r in profiles),target_allocations=allocations)
(base/'poststudy.json').write_text(json.dumps(result,indent=2))
report=(base/'RESULTS.md').read_text()
report=report.replace('A default recommendation requires complete validated results.',
    'The completed study supports the automatic multiplication choice for rank paths. Retain all fitting/stopping/rank/DF tolerances and the dense diagnostic fallback.')
report+='''

## Decision and limits

Retain the six-line candidate as an automatic internal multiplication policy.
All six full rank paths have paired intervals below one; the primary
many-response path ratio is0.781 (95%0.761–0.838), and wide is0.905
(0.901–0.912). Individual wide, ill-conditioned and rank-deficient fits
also improve with intervals below one. Small, medium and many-response
single-fit intervals cross one: their speed benefits are uncertain. The
many-response single-fit paired median is0.891 (0.832–1.052), with point
medians6.821→6.462ms. Do not turn that into a precise single-fit speedup.
Every individual input's ratio and any reversals remain in summary.json.

The historical SciPy coefficient-only gap does not persist on this EPYC7513
CPU and current stack, even for the fresh baseline. This is evidence about
these implementations/environments and outputs, not a language ranking or a
gap closed entirely by this change. Coefficient-only code is unchanged;
its small before/after timing movements are a negative control. Native R
QR/SVD is slower at the tested point medians, but returns a different result
contract and may choose non-minimum-norm coefficients on nonunique designs.

Cached many-response residual diagnostics measure roughly2.5–2.7ms for the
dense product and1.6–1.7ms for the thin product. Decompositions, centering,
rank fallback and output construction remain substantial. These independent
cached timings are not an additive explanation of complete-fit latency;
public ratios above provide the adoption evidence. In-place buffers or
additional cross-solution caches were not introduced: the path already shares
the design/response decomposition, and this measured change reuses its factors.
Further cache changes would need their own evidence.

No uniform memory reduction is established. Many-response timing-peak RSS
medians are435.5/436.7MiB (baseline/candidate); wide whole-worker peaks are
567.5/587.9MiB and include separate component/profiling work. Julia cumulative
allocations and all current/peak measurements are retained separately. A
three-round warm process study does not establish startup latency, performance
on all hardware, prediction risk, rank-selection quality or inference accuracy.

All480 measured processes completed, with5040 timing rows,7200 component
rows,440 CPU and80 allocation reports. All4200 independent output certificates
passed; all1200 baseline/candidate coefficient pairs are bitwise equal.
The targeted source tests passed720 checks on each Julia1.10/1.13, including
Float32/64, zero/near-threshold, wide/rank-deficient and golden controls.
Final-head hosted CI is recorded separately in PR metadata.

Versions: Julia1.13.2, Python3.11.9, NumPy2.4.6, SciPy1.17.1, R4.5.2,
rrpack0.1-14. Julia uses ILP64 OpenBLAS; NumPy reports OpenBLAS0.3.31.188.0
and SciPy0.3.30, both Haswell kernels and one thread; R uses OpenBLAS0.3.30.
Full per-worker metadata, R sessions, frozen Julia manifests, input seed/hash
records, process outcomes, profiles and exact-source checks accompany the
tables. Source snapshots and raw fitted arrays stay in the owning Discovery
run. Harnesscc02719 was frozen before fresh evaluation inputs1–20; the
poststudy reads saved records only. Recreate with the README and launchers.
Pilot failures and scoped recovery are preserved and excluded from evaluation.
'''
(base/'RESULTS.md').write_text(report)
print(json.dumps(result,indent=2))
