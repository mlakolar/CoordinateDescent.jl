# Primal–dual quantile API: decision and measurements

**Use warm-started HiGHS dual simplex for exact fits and research duals.**
The API keeps raw optimizer scores at every saved penalty, including at zero
residuals. Feasible gap certificates are stored separately.

```julia
using HighDimensionalModels, HiGHS
path = QuantileLassoPath(X, y, [0.2, 0.1, 0.05])
u = dual_scores(path)  # observations × penalties
xi = -u / nobs(path)   # QR.jl convention: -u/n
```

Mean seconds per complete **32-level exact path**, including duals and
diagnostics. Each numeric entry passes all six measurements:

| Input | Exact API | Gurobi primal simplex | quantreg + native scores¹ |
|---|---:|---:|---:|
| 1,000 × 100 | 0.239 | 0.599 | 0.984 |
| Tail τ=0.1, 1,000 × 100 | 0.174 | 0.503 | 1.035 |
| 10,000 × 100 | 2.749 | 40.13 | 13.84 |
| Correlated 1,000 × 1,000 | 14.76 | 15.86 | No qualified path |

The wide API/Gurobi difference is inconclusive: its paired 95% interval is
[−0.80, 3.01] seconds. Neither exact backend completes the 1,000 × 3,000
16-level stress path within 60 seconds. Check `complete` and `converged`;
a stopped fit can retain an infeasible diagnostic iterate.

**Smoothing is fast, but equal-dual-accuracy speed claims are premature.**
The API passes only 4/6 wide paths at default settings. Tighter controls pass
2/2 on the failed input at 4.48 versus 1.77 seconds, yet still miss the stricter
2e−6 gate. Small/tall tight controls also miss that gate; conquer's 1e−7 setting
passes. On the 3,000-column stress case the API passes 6/6 and conquer 2/6.
Smoothed scores are not automatically interchangeable with exact inference scores.

¹quantreg is adapted to retain native workspace scores. rqPen BR/FN pass
coefficient checks on small/tail/tall, but the tested interfaces expose no duals.
Huber solves another approximation.

Discovery completed **240 primary observations, 28 separate accuracy controls
and eight CPU profiles**, using three inputs and two shuffled repetitions.
This phase used 84m27s of running allocation time and 2.65 CPU-hours, with at most
2 CPUs / 16 GiB. [Full intervals and qualification counts](results/discovery-2026-10-25/summary/TABLE.md)
and the [recreation record](results/discovery-2026-10-25/run-record.json) retain failures.

**Validation:** 3,089 package assertions passed, including 592 API checks;
the renamed API also passes all 592 on Discovery Julia LTS. Hosted Julia tests,
docs and examples pass at the integration commit. Codecov's project check flags
99.53% versus 99.88% (−0.35 percentage points); its patch check passes.

Next: refine smoothed stopping, then pivotal calibration, CV and inference.
Full parametric breakpoint enumeration remains deferred. [API documentation](../../../docs/src/quantile.md).
