import csv, json, pathlib, statistics
root=pathlib.Path(__file__).resolve().parent
out=root/'results'
summary=json.loads((out/'summary.json').read_text())
rows=summary['table']
resources={v:[json.loads(p.read_text()) for p in sorted(out.glob(f'resource-{v}-*.json'))] for v in ('baseline','candidate')}
assert all(len(z)==7 and all(x['exit_code']==0 for x in z) for z in resources.values())
profiles=list(out.glob('cpu-*.txt'));allocs=list(out.glob('alloc-*.txt'))
assert len(profiles)==len(allocs)==112
assert all(int(p.read_text().splitlines()[0].split('=')[1])>0 for p in profiles)
lines=['# Structured degrees of freedom: Discovery evaluation, 2026-10-26','',
'The component-basis and path-cache implementation reduces complete path time in this fixed evaluation. Every compared coefficient, objective, iteration count and degree-of-freedom value is identical. These results concern diagnostic implementation cost; they do not change or calibrate a statistical estimator.','',
'## Complete public paths','',
'Seven independent datasets per design, three warmed rounds per dataset. Times are medians of within-dataset medians. Ratios and intervals use paired candidate/baseline dataset ratios; the intervals are percentile-bootstrap intervals for their median (10,000 resamples, seven independent units). Smaller ratios mean less candidate time. All imports, compilation, input loading, independent checks and profiling are outside fit timers; model setup and required public diagnostics are inside.','',
'| Case | Baseline ms | Candidate ms | Paired ratio | 95% interval | Paired range | Baseline allocated MiB | Candidate allocated MiB |',
'|---|---:|---:|---:|---|---|---:|---:|']
for r in rows:
    if r['kind']!='path': continue
    lo,hi=r['ratio_ci95']
    lines.append(f"| {r['case']} | {1000*r['baseline_seconds']:.3f} | {1000*r['candidate_seconds']:.3f} | {r['paired_median_ratio']:.3f} | {lo:.3f}–{hi:.3f} | {r['ratio_min']:.3f}–{r['ratio_max']:.3f} | {r['baseline_bytes']/2**20:.2f} | {r['candidate_bytes']/2**20:.2f} |")
lines += ['', 'Allocated bytes are cumulative allocation, not peak resident memory. The individual dataset ratios, every timing round and allocated-byte observations are retained in summary.json and timings-*.csv. The seven-input intervals describe this experiment; they do not establish a universal speed ranking.','',
'## Degrees-of-freedom work and setup','',
'These are separate diagnostic measurements on the returned path coefficients, with a fresh one-pattern cache per round. They are not additive decompositions of the public-path timings. Baseline diagnostic calls have no reusable design-rank preparation. Candidate preparation includes the design SVD and row-oriented penalty copy; this cost is included in public-path timings above.','',
'| Case | Baseline DF ms | Candidate DF ms | Candidate setup ms | Distinct patterns |',
'|---|---:|---:|---:|---:|']
raw=[]
for p in sorted(out.glob('timings-*.csv')):
    raw.extend(csv.DictReader(p.open()))
for r in rows:
    if r['kind']!='df':continue
    setup=next(z for z in rows if z['case']==r['case'] and z['kind']=='df_setup')
    counts=sorted(set(int(x['pattern_count']) for x in raw if x['case']==r['case'] and x['kind']=='df'))
    lines.append(f"| {r['case']} | {1000*r['baseline_seconds']:.3f} | {1000*r['candidate_seconds']:.3f} | {1000*setup['candidate_seconds']:.3f} | {min(counts)}–{max(counts)} |")
r=next(z for z in rows if z['kind']=='df_including_setup')
lines += ['',f"The p=1000 diagnostic, including one workspace setup and one DF evaluation, took {1000*r['baseline_seconds']:.3f} ms at baseline and {1000*r['candidate_seconds']:.3f} ms with the candidate; paired median ratio {r['paired_median_ratio']:.3f}, interval {r['ratio_ci95'][0]:.3f}–{r['ratio_ci95'][1]:.3f}, range {r['ratio_min']:.3f}–{r['ratio_max']:.3f}. This is a single diagnostic workload, not a public p=1000 path measurement or a large-signal CV claim.",'',
'The retained cache owns a sparse row representation of the fixed penalty and one inactive-row pattern, with storage proportional to penalty nonzeros and row count. Orthonormal component columns avoid dense null-space storage for graph/pin constraints. A rank-deficient design still requires rank(X*C); counting components alone would be incorrect. General contrasts and ambiguous rank/weight cases retain dense fallback work.','',
'## Correctness, profiles and memory','',
(out/'validation.txt').read_text().strip(),'',
'This is 56 shared input/design cases and 1,820 paired path levels (3,640 returned fits). All final fits converged. All 1,050 timing rows are present. Shared input digests agree between implementations. Final Julia 1.10.12 full suite: 5,877/5,877; Julia 1.13.2: 5,873/5,873. Both include independent dense-null-space graph/contrast/rank checks, the rectangular numerical-rank regression and existing R golden paths.','',
'There are 112 CPU profile reports and 112 allocation reports: eight cases × seven independent shared inputs × two implementations, separate from timings. CPU reports contain nonempty sampled stacks; raw stack-word counts are not wall-time percentages. The directly timed DF reduction and unchanged solver iterations support the source-level explanation: fewer repeated dense null-space/rank calculations, with reusable design checks and constraint patterns. Do not sum inclusive profiler frames or infer exact causal percentages from them.','',
'| Fresh worker | Median peak MiB | Range MiB |','|---|---:|---:|']
memory={}
for v,z in resources.items():
    peaks=[x['peak_rss_bytes']/2**20 for x in z]
    memory[v]=dict(median_peak_mib=statistics.median(peaks),min_peak_mib=min(peaks),max_peak_mib=max(peaks))
    lines.append(f'| {v} | {statistics.median(peaks):.1f} | {min(peaks):.1f}–{max(peaks):.1f} |')
lines += ['',
'Each worker is a fresh process for one implementation and dataset replication, covering all eight designs and the p=1000 diagnostic. RSS includes imports/JIT, independent checks and separate profiles; it is neither optimizer workspace nor a per-design/per-fit memory bound. metadata-*.txt retains post-warm current RSS by case and final peaks. resource-*.json records child-process accounting.','',
'The resident-memory ranges overlap, and the candidate has the larger maximum worker peak. No uniform peak-memory reduction is established.','', '## Defaults and scope','',
'Use the component calculation and bounded path cache automatically for the supported constraints: complete public-path gains, unchanged results and reduced allocations support that choice on these cases. Keep the dense fallback for general contrasts and numerical ambiguity. Retain df_tol=1e-4 and the existing family/precision-specific stopping settings; this experiment provides no statistical reason to retune them. DF near fusion/rank transitions remains tolerance-sensitive. Generic trend cases, rank-deficient and wide designs remain in the tables. The cache setup has a cost, so do not extrapolate these multi-level path gains to arbitrary one-off problems.','',
'No penalty-selection, coverage, prediction-risk or FDR improvement is claimed. Large signal cross-validation remains issue100. This change does not establish high-dimensional scaling beyond the measured path dimensions.','',
'## Source and recreation','',
'Baseline: master 295fbf5420d6384a1ca261023367ee86d0a40b8c. Measured candidate production: 49289d477850cd8eeb052b39cb4af79e1b08d2fe. Candidate structured.jl SHA256:8bd9c38fcb4f183815163b1acccafb858b68a5247c35d4ed3d789983ff7a19fb; baseline:05bd75aa15fe19aa5ef7def00c0a8bc229cdf58779b4320884e8e040d0d8ae9d. Source hashes are emitted by every worker. Dependencies are held fixed: PB b6c949f3dd26cad09252e1c2c9e49810eb69403e, PS 531ff55fcf8a5ce2c6997c144d4f57e85cf79f3e, KS 1feff286d0020764d2c1b03d733ab316cf948ead. Resolved manifests and snapshots remain with the run.','',
'Discovery mkolar_1314/main, EPYC7513, one Julia/BLAS/OpenMP/MKL thread per worker, Julia1.13.2 for timing; one 8CPU/32GiB allocation, seven exclusive workers. Alternate baseline/candidate process order by replication. Tests: job13447103; evaluation/validation/aggregation:job13447106. Seed-zero calibration is separate from final seeds1–7. No candidate policy was tuned on final outcomes.','',
'See the [harness README](../../README.md) and executable scripts. Actual remote run root:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99; final tree is candidate2/ with env-candidate2-1.13; baseline is baseline/ with env-baseline-1.13. Launch: sbatch --parsable --dependency=afterok:13447103 measure-v2.sbatch. The latter uses worker-v2.sh and the same committed bench/validate/summarize sources. Initial launcher and incomplete-source staging attempts are preserved in the task registry; they do not contribute timings here. All final reporting inputs, failures, profiles and accounting are retained. Human review is required; no merge performed.','']
(out/'RESULTS.md').write_text('\n'.join(lines))
(out/'resource-summary.json').write_text(json.dumps(dict(memory=memory,cpu_profiles=len(profiles),allocation_profiles=len(allocs)),indent=2))
print('\n'.join(lines))
