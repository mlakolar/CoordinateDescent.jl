# Fixed-rank residual construction: Discovery study

Phase: main. 20 independent shared inputs in each of six designs, three warmed rounds per method/input. The seed-zero pilot is separate from evaluation inputs1–20. No policy is retuned on evaluation outcomes.

Complete native public fits/paths are separate from coefficient-only controls. The SciPy full-output adapter returns equivalent coefficients, residuals, intercepts, rank, support, objective and both DF definitions, using thin factors with the same conservative dense rank fallback. R is rrpack's native QR/SVD interface; its output contract differs and its wide/rank-deficient coefficients need not be minimum norm. Invalid R normal-equation shortcuts are excluded.

## Public API comparison

| Case | Mode | Baseline Julia ms | Candidate Julia ms | Candidate/baseline (95% interval) | SciPy full-output ms | R QR/SVD ms |
|---|---|---:|---:|---|---:|---:|
| small | public_fit | 1.024 | 0.971 | 0.983 (0.970–1.008) | 1.225 | 1.490 |
| small | public_path | 1.111 | 1.058 | 0.951 (0.938–0.985) | 1.748 | 1.602 |
| medium | public_fit | 1.864 | 1.814 | 0.973 (0.952–1.010) | 2.200 | 5.173 |
| medium | public_path | 2.352 | 2.032 | 0.901 (0.857–0.915) | 3.096 | 6.032 |
| responses | public_fit | 6.821 | 6.462 | 0.891 (0.832–1.052) | 8.180 | 71.875 |
| responses | public_path | 14.728 | 11.983 | 0.781 (0.761–0.838) | 13.398 | 104.577 |
| wide | public_fit | 12.962 | 12.666 | 0.971 (0.967–0.984) | 14.232 | 17.049 |
| wide | public_path | 14.625 | 13.210 | 0.905 (0.901–0.912) | 14.834 | 16.464 |
| ill | public_fit | 4.945 | 4.833 | 0.966 (0.951–0.977) | 5.264 | 11.026 |
| ill | public_path | 6.312 | 5.600 | 0.890 (0.860–0.905) | 6.279 | 11.859 |
| deficient | public_fit | 2.144 | 2.076 | 0.950 (0.924–0.976) | 2.411 | 6.516 |
| deficient | public_path | 2.754 | 2.344 | 0.823 (0.814–0.867) | 3.361 | 6.309 |

Times summarize within-input medians. Ratios use paired input medians; intervals bootstrap the median of independent input ratios (10,000 resamples). Ranges and every ratio are retained. These are warm-process, newly constructed fits, not process-startup latencies. Imports/JIT/loading/checks/profiles are outside timers; centering, decomposition and required outputs are inside public timers.

## Coefficient-only controls

| Case | Mode | Julia baseline ms | Julia candidate ms | SciPy ms |
|---|---|---:|---:|---:|
| small | coefficient_fit | 0.934 | 0.916 | 0.962 |
| small | coefficient_path | 0.936 | 0.922 | 0.947 |
| medium | coefficient_fit | 1.539 | 1.510 | 1.580 |
| medium | coefficient_path | 1.551 | 1.552 | 1.575 |
| responses | coefficient_fit | 4.351 | 4.318 | 5.564 |
| responses | coefficient_path | 4.473 | 4.509 | 5.344 |
| wide | coefficient_fit | 12.503 | 12.412 | 13.843 |
| wide | coefficient_path | 12.691 | 12.738 | 13.611 |
| ill | coefficient_fit | 4.607 | 4.650 | 4.716 |
| ill | coefficient_path | 4.685 | 4.666 | 4.727 |
| deficient | coefficient_fit | 1.948 | 1.962 | 2.125 |
| deficient | coefficient_path | 1.970 | 2.001 | 2.069 |

Cached-input component timings are in components.csv. They are not an additive decomposition of public runtime. CPU profiles cover every independent many-response/wide input for baseline, candidate, SciPy and R; allocation profiles are separate from timing. Profile inclusive frames are not additive wall-time percentages.

## Correctness and memory

```json
{
  "shared_cases": 120,
  "output_records": 4200,
  "expected_output_records": 4200,
  "failed_certificates": 0,
  "paired_records": 1200,
  "expected_paired_records": 1200,
  "failed_pairs": 0,
  "bitwise_equal_pairs": 1200,
  "errors": []
}
```

Independent certificates check closed-form loss, feasible rank, predictions, minimum-norm representatives where required, and all native/SciPy returned residual/intercept/support/DF fields. R nonunique coefficients are judged through identifiable fitted values; coefficient/minimum-norm differences remain in certificates.csv. Every baseline/candidate coefficient pair is checked directly.

memory.csv records fresh process per input and implementation, current/peak RSS after warming, peak after timing and whole-worker peak. Workers cover the four modes (two for R); RSS includes imports/JIT/data/output persistence and whole-worker peaks also include separate components/profiles. It is not optimizer workspace or a single-fit memory bound. Julia cumulative allocated bytes are reported separately in aggregate.csv.

## Interpretation and recreation

Do not adopt the candidate from inner-kernel savings alone. Judge public fit/path paired ratios and their uncertainty, preserve small or reversed effects and any remaining SciPy gap, and retain the existing design-rank cutoff and diagnostic fallbacks. The predeclared multiplication choice is rank*(p+q)<p*q; this study changes no statistical objective, rank-selection rule or tolerance. The completed study supports the automatic multiplication choice for rank paths. Retain all fitting/stopping/rank/DF tolerances and the dense diagnostic fallback.

Pinned baseline:295fbf5420d6384a1ca261023367ee86d0a40b8c. Candidate production:9dd42f86f4a89c947b8205407de065cf12b9fbbf. Frozen PB/PS/KS and Julia environments are recorded in the run registry and worker metadata. Timing Julia1.13.2, Discovery EPYC7513, one Julia/BLAS/OpenMP/MKL thread. All numerical generation, fitting, profiling, validation and aggregation run under Slurm. Eight exclusive workers within8CPUs/32GiB. See README and executable launchers; no merges are authorized.


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
