# Smoothed quantile paths: stopping repair and focused comparison

**The new stopping rule resolves the observed raw-dual accuracy problem.**
Proximal Newton still uses coordinate descent, backtracking, screening and warm
starts. Convolution conjugates now enable gap stopping; acceptance also checks
the actual scores and intercept constraint. Damping and absolute inner stopping
avoid cancellation in flat kernel tails. `dual_scores` retains the raw scores;
feasible certificates remain separate. Exact HiGHS fits are unchanged.

Discovery completed **130 primary path measurements and 16 separate controls**.
Each path has 32 penalties. Both new tolerance settings pass the common gate on
all 26 primary attempts; legacy objective stopping passes 0/26. The tighter
setting also passes the stricter 2e−7 gate on all 26.

Mean seconds including scores and diagnostics, three fresh inputs and two
shuffled repetitions per input; every table entry qualifies at every penalty:

| Input | New API, tolerance 2e−7 | conquer | conquer setting |
|---|---:|---:|---:|
| 1,000 × 100 | 0.082 | 0.176 | 1e−7 |
| Tail τ=0.1, 1,000 × 100 | 0.177 | 1.625 | 1e−7 |
| 10,000 × 100 | 0.654 | 0.638 | 1e−6 |
| Correlated 1,000 × 1,000¹ | 3.996 | 33.81 | 1e−8 |

¹Wide timings come entirely from the separate tighter-reference control:
conquer at 1e−7 qualifies in only 2/6 fresh primary attempts. Its survivor mean
is not ranked. The control's paired 95% interval for time saved is
[28.65, 30.98] seconds. Tall performance is comparable at the faster qualifying
conquer setting. These are focused comparisons, not a universal speed claim.

The common gate is KKT ≤2e−6 and absolute raw gap ≤2e−6(1+|objective|).
Two post-timing CPU profiles retain the coordinate-update and matrix-work costs.
Inner warnings and conservative native convergence flags remain visible even
when the returned pair passes the external gate. Smoothed scores still require
statistical justification before use in exact-quantile inference.

**Validation:** 2,232 conjugate/Fenchel, 360 Newton and 40 edge-case assertions;
615 API assertions with the new backend and 583 with the older backend; relevant
GLM, gap and Newton regressions all passed. Two failed development canaries are
retained. This stage used 17m14s of running allocation time, at most 2 CPUs /
16 GiB; the combined API campaign used 102m20s of its 120-minute allowance.

[Full primary table](results/discovery-2026-10-25/gap-main/summary/TABLE.md),
[wide control](results/discovery-2026-10-25/gap-main/tight-reference/summary/TABLE.md),
[recipe](README.md) and [run record](results/discovery-2026-10-25/run-record.json).
The subsequent [calibration and CV implementation](../tuning/RESULTS.md) is
tracked separately from these measurements. Inference is now
[issue #104](https://github.com/mlakolar/HighDimensionalModels.jl/issues/104);
full parametric breakpoints remain deferred.
