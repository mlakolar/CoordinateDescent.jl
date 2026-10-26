"""Discovery compute-node checks/report from saved Gaussian fits; no refitting."""
import csv,json,math,re,statistics
from pathlib import Path
D=Path('reports/campaign-gaussian')
rows=list(csv.DictReader((D/'records.csv').open()));di=list(csv.DictReader((D/'diagnostics.csv').open()))
assert len(rows)==100000 and len(di)==100000
by={};maxp=maxinterval=0.
for r in rows:
    assert r['converged'].lower()=='true'
    b,s=float(r['estimate']),float(r['se'])
    maxp=max(maxp,abs(float(r['pvalue'])-math.erfc(abs(b/s)/math.sqrt(2))))
    maxinterval=max(maxinterval,abs(float(r['lower'])-(b-1.959963984540054*s)),abs(float(r['upper'])-(b+1.959963984540054*s)))
    key=(r['case'],int(r['rep']));by.setdefault(key,[]).append(r)
assert len(by)==10000 and all(len(v)==10 for v in by.values()) and maxp<1e-12 and maxinterval<1e-12
hashes={}
for r in di:
    key=(r['case'],int(r['rep']));pair=(r['input_x_sha256'],r['input_y_sha256'])
    assert all(re.fullmatch('[0-9a-f]{64}',h) for h in pair)
    assert key not in hashes or hashes[key]==pair
    hashes[key]=pair
    assert float(r['noise_ratio'])<=1.5*(1+1e-8)
assert len(hashes)==10000
# Distinct data hashes across datasets; within-dataset hashes must match all ten targets.
assert len(set(hashes.values()))==10000
checks=dict(datasets=len(by),target_rows=len(rows),diagnostic_rows=len(di),max_normal_pvalue_error=maxp,max_interval_error=maxinterval,unique_dataset_hash_pairs=len(set(hashes.values())),hashes_consistent_within_dataset=True,note='Hash format, uniqueness and within-dataset consistency checked; exact regeneration parity remains a requirement for competitor arms.')
(D/'additional-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
fits=[]
for case in sorted({r['case'] for r in rows}):
    for phase in ['first_per_worker','warm']:
        v=[float(v[0]['seconds']) for (c,rep),v in by.items() if c==case and ((rep<=8)==(phase=='first_per_worker'))]
        fits.append(dict(case=case,phase=phase,datasets=len(v),mean_seconds=statistics.mean(v),mcse_seconds=statistics.stdev(v)/math.sqrt(len(v)),median_seconds=statistics.median(v),max_seconds=max(v)))
(D/'fit-time-summary.json').write_text(json.dumps(fits,indent=2)+'\n')
rr=[]
for p in sorted(Path('resources').glob('gaussian-campaign-main-*.txt')):
    r=dict(line.split('=',1) for line in p.read_text().splitlines() if '=' in line)
    assert all(k in r for k in ['cpu_seconds','process_seconds','max_rss_kib','process_exit'])
    assert float(r['process_exit'])==0
    rr.append({k:float(v) for k,v in r.items()})
assert len(rr)==80
resource=dict(workers=len(rr),cpu_seconds=sum(r['cpu_seconds'] for r in rr),worker_process_seconds=sum(r['process_seconds'] for r in rr),max_worker_rss_gib=max(r['max_rss_kib'] for r in rr)/1024**2,median_worker_rss_gib=statistics.median(r['max_rss_kib'] for r in rr)/1024**2,note='Per-worker Linux child-process peak RSS, not simultaneous total. Worker processes include startup/import/compilation and 125 datasets each. Allocation: 64 CPUs,128 GiB; elapsed1426s. Fitting timer includes initial CV, projection, HC3 but excludes diagnostic checks and file I/O. First per worker includes fitting compilation after imports, not full fresh-process startup.')
(D/'resource-summary.json').write_text(json.dumps(resource,indent=2)+'\n')
ss=json.loads((D/'strict-summary.json').read_text());slots=json.loads((D/'per-slot.json').read_text());gates=json.loads((D/'calibration-gates.json').read_text())
flags=[]
for r in slots:
    active=float(r['truth'])!=0
    lower=r['coverage']-1.959963984540054*r['coverage_mcse'];upper=r['rejection']+1.959963984540054*r['rejection_mcse']
    if (active and lower<.925) or (not active and upper>.075):flags.append(dict(**r,coverage_lower95_mc=lower,rejection_upper95_mc=upper))
(D/'role-review.json').write_text(json.dumps(dict(unadjusted_role_flags=flags,role_coverage_range=[min(r['coverage'] for r in slots if float(r['truth'])!=0),max(r['coverage'] for r in slots if float(r['truth'])!=0)]),indent=2)+'\n')
lines=['# Gaussian production-policy arm: 1,000 fresh datasets per design','','EMPIRICAL. Frozen Gaussian HC3 policy; 10 designs, 10,000 response datasets and 100,000 marginal coefficient intervals. All fits were complete, finite and converged. This is the own-policy arm; the expanded competitor comparison is not complete. The policy was frozen before the independent 500-per-design confirmation and was not retuned for this campaign.','','|Design|Active coverage ± MCSE (%)|Null type I ± MCSE (%)|Power ± MCSE (%)|Active interval length ± MCSE|Warm fit mean (s)|','|---|---:|---:|---:|---:|---:|']
for c in sorted({r['case'] for r in ss}):
    a=next(r for r in ss if r['case']==c and r['target_class']=='active');n=next(r for r in ss if r['case']==c and r['target_class']=='null');t=next(r for r in fits if r['case']==c and r['phase']=='warm')
    lines.append(f"|{c}|{100*a['coverage']:.2f} ± {100*a['coverage_mcse']:.2f}|{100*n['rejection']:.2f} ± {100*n['rejection_mcse']:.2f}|{100*a['rejection']:.2f} ± {100*a['rejection_mcse']:.2f}|{a['length']:.4f} ± {a['length_mcse']:.4f}|{t['mean_seconds']:.3f}|")
lines+=['',f"All {sum(g['engineering_calibration_gate'] for g in gates)}/10 pooled engineering calibration gates passed. There are {len(flags)} unadjusted target-role gate flags; see role-review.json and per-slot.json. These are finite-design experiments, not proof of uniform nominal coverage.",'',f"Job 13450404: 1,426 seconds elapsed; 64 CPUs and 128 GiB allocated. Consumed worker CPU {resource['cpu_seconds']/3600:.2f} CPU-hours. Maximum observed worker RSS {resource['max_worker_rss_gib']:.3f} GiB; median {resource['median_worker_rss_gib']:.3f} GiB. These are per-process peaks, not total simultaneous job memory.",'','Monte Carlo errors use independent datasets as clusters, pooling five active or five null roles within each dataset. Per-role results and complete/planned counts are retained. Intervals are marginal normal intervals; simultaneous and bootstrap coverage were not evaluated. Correlation and weak signals must be assessed jointly with power and interval length. Refitting is not part of this frozen Gaussian policy because it worsened several development designs. Logistic and QP policies remain experimental.','','Recreation: source gaussian-campaign-v1, package-hc3-v1/env-hc3-v1, campaign namespace 92300000, ten prespecified cases; campaign-gaussian.jl with eight shards/case, reps1–1000. analyze-gaussian-campaign-v2.py campaign 1000 then this script on a Discovery compute node. Input Float64-byte SHA256 pairs are retained in diagnostics.csv for exact regeneration in competitor arms.','']
(D/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(dict(checks=checks,resources=resource,pooled_gates_passed=sum(g['engineering_calibration_gate'] for g in gates),role_flags=len(flags))))
