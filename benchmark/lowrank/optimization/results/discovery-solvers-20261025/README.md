# PANOC, block FISTA, and fixed-rank diagnostic reuse

This follow-up implements optional PANOC for coefficient nuclear-norm regression,
weighted block FISTA for row-sparse reduced-rank regression, and cheaper
fixed-rank result diagnostics. APG and coordinate descent remain the defaults.
Both solver choices propagate through paths and cross-validation.

```julia
using CoordinateDescent, ProximalAlgorithms
nuclear_regression(X, Y, λ; solver = :panoc)
sparse_reduced_rank(X, Y, r, λ; block_solver = :fista)
```

All tests, numerical measurements, profiles, and numerical aggregation ran in
Discovery Slurm allocations. The user selected a fixed Intel Xeon Silver 4116
CPU model in the debug partition after the main partition reached its running
job limit. Every competitor and both package versions use this CPU model.
Absolute timings should not be compared directly with the earlier EPYC 7513
campaign. MATLAB is excluded. Statistical performance is not evaluated.

The six designs and their five independent shared inputs are unchanged:
small (300,50,10), medium (300,50,100), many responses (300,50,1000), wide
(200,400,100), ill-conditioned (500,100,100; condition number 1e4), and
rank-deficient (300,80,100; design rank 40). Tuples denote (n,p,q), with
working rank 3. Each input has three timing rounds in rotating method order.
Tables summarize the five within-input medians, in milliseconds. Paired
ratios compare the same input; their medians need not equal ratios of table
medians. All workers use one CPU and one Julia/BLAS/OpenMP thread.

The nuclear penalty is on B: `||Yc-Xc B||²_F/(2n) + λ||B||_*`.
All nuclear alternatives solve this same convex objective. Native PANOC uses
ProximalBase's workspace-based nuclear prox and the standard PANOC step
0.95/L; the generic ProximalAlgorithms/ProximalOperators adapter retains 1/L.
The row-sparse problem alternates weighted group-lasso predictor updates and
an orthogonal Procrustes response update. Nonconvex stationarity certifies a
local solution; attained objective differences are retained.

End-to-end public fits include centering, initialization, and result diagnostics.
Labels `cd_sparse` and `block_fista` denote the older shared-initialization
kernel adapters; `cd_sparse_api` and `cd_sparse_fista` include public API
initialization and diagnostics. Before/after rows compare public APIs on both
sides. Compilation, input loading, certificates, factor replay, and profiles
are outside fit timers. Fresh-process high-water RSS includes the runtime,
packages, warm-up, and post-fit certificate/factor-replay work, with a separate
post-warm-up baseline; it is not an allocation count for one solver call.

At the middle penalty, native PANOC takes **14–38% less time on five of six
designs**; it is about 4% slower on the many-response design. Fixed-rank public
fits take **17–39% less time**, with identical attained objectives. Rank paths
take **47–60% less time**. Public block FISTA has its clearest gains on
ill-conditioned and rank-deficient inputs; its gains elsewhere are small once
initialization and result diagnostics are included.

| Case | Previous rank API | Updated rank API | SciPy coefficients | R QR/SVD | Updated/previous paired time |
|---|---:|---:|---:|---:|---:|
| Small | 2.93 | 2.11 | 2.32 | 5.23 | 0.715 |
| Medium | 5.36 | 4.27 | 3.68 | 15.85 | 0.801 |
| Many responses | 18.33 | 15.30 | 9.80 | 156.78 | 0.828 |
| Wide | 43.83 | 29.78 | 28.77 | 47.10 | 0.680 |
| Ill-conditioned | 17.64 | 12.85 | 11.16 | 29.17 | 0.689 |
| Rank-deficient | 7.78 | 4.75 | 4.73 | 17.68 | 0.611 |

The updated/previous paired ratios for rank paths 0–3 are 0.473, 0.535,
0.513, 0.426, 0.491, and 0.400 in table order. Reusing the design SVD and
computing coefficient rank through the small factor core avoids repeated
large decompositions. Ambiguous numerical-rank cases retain a dense fallback;
rank, residuals, support, and degrees-of-freedom semantics are tested. SciPy
returns coefficients alone; our public API still computes diagnostics. The
many-response public API remains slower than that coefficient-only comparator.

| Case | Native APG | Native PANOC | Generic Julia PANOC | Python FISTA | Python ADMM |
|---|---:|---:|---:|---:|---:|
| Small | 4.69 | 4.04 | 3.32 | 16.47 | 17.70 |
| Medium | 61.68 | 42.72 | 40.17 | 132.44 | 121.19 |
| Many responses | 313.10 | 323.48 | 318.16 | 696.90 | 664.34 |
| Wide | 876.11 | 565.53 | 595.85 | 1920.71 | 1762.80 |
| Ill-conditioned | 340.58 | 231.61 | 188.51 | 968.71 | 882.65 |
| Rank-deficient | 71.71 | 55.16 | 50.96 | 139.53 | 149.87 |

Generic PANOC remains faster than the native wrapper on five of six designs
at this penalty; the native option wins on the wide design. The generic
adapters return coefficients without the full native result diagnostics, and
the PANOC adapters also differ in prox implementation and step size. These
results do not isolate programming-language overhead. The native APG
before/after baselines are essentially unchanged.

For eight descending nuclear penalties, native PANOC/APG paired ratios are
0.959, 0.888, 1.160, 0.605, 0.602, and 1.046 in table order. PANOC therefore
helps four of six paths, with the largest gains on wide and ill-conditioned
inputs. The APG default is retained. Path certificates cover the last point;
native flags are the conjunction across all eight levels. Independent saved
coefficient checks cover cold fits and rank-path endpoints, rather than every
intermediate nuclear-path point.

| Case | Public coordinate descent | Public block FISTA | R rrpack srrr | FISTA/CD paired time |
|---|---:|---:|---:|---:|
| Small | 3.34 | 3.26 | 8.95 | 0.967 |
| Medium | 6.77 | 6.62 | 11.91 | 0.973 |
| Many responses | 27.71 | 27.50 | 53.47 | 1.001 |
| Wide | 70.37 | 66.87 | 47.89 | 0.951 |
| Ill-conditioned | 83.95 | 32.46 | 231.64 | 0.384 |
| Rank-deficient | 12.56 | 9.44 | 30.48 | 0.754 |

Public block FISTA reduces time by about 62% on ill-conditioned inputs and
25% on rank-deficient inputs. Elsewhere its paired ratio is 0.951–1.001.
The shared-initialization kernel experiment shows larger inner-solver gains
on medium and many-response cases, but these do not translate into comparable
public API speedups. For example, the many-response kernel takes 18.46 ms
with coordinate descent and 6.95 ms with generic block FISTA, while both public
APIs take about 27.5 ms. Initialization, result diagnostics, and differences
in numerical trajectories matter.

R is faster than both public sparse APIs on the wide design at the middle
penalty, with matching attained objectives. At the strongest wide penalty,
R reaches a worse stationary point on four of five inputs, with a maximum
relative objective difference of 0.163; that speed advantage is not an
equivalent-objective result. All native CD/FISTA attained objectives agree
within numerical precision across the three tested penalties.

The five-input adaptive-weight supplement uses common inverse-row-norm-squared
weights. Its public FISTA/CD paired ratio is 1.032, ranging from 0.974 to 1.046;
there is no consistent public API timing gain on these inputs. All 30 saved
weighted fits pass the actual-factor stationarity checks. The sparse-component
coordinate solver remains faster than both tested component alternatives on
all six middle-penalty designs; see the full table for every method and penalty.

There are **7,110 distinct timing observations**, **430 text profile reports**
(535 profile files including Python/R raw formats), and **1,860 independent
saved-fit checks**. Each timing group contains three rounds on each of all
five independent inputs. **1,855 checks pass**. The only failures are the five
known ill-conditioned R normal-equation fits, producing 15 failed timing rows;
the separate R QR/SVD route passes all 30 inputs. All saved native fits pass,
and all native timed fits report convergence. Nuclear checks require both
relative dual gap and normalized proximal-gradient mapping <=1e-6; sparse
checks use actual A/V factors, predictor KKT, response stationarity, and
orthogonality. These are optimization checks, not statistical comparisons.

Profiles on all five medium, many-response, and wide inputs show that LAPACK
SVD and BLAS multiplication remain major costs for fixed-rank and nuclear
fits. Native PANOC repeatedly calls the ProximalBase nuclear workspace. Sparse
public fits also retain SVD, eigensolver, and allocation costs, which helps
explain why inner-block improvements can have modest overall impact. Python
profiles concentrate in SciPy's SVD; R component profiles primarily resolve to
`.Call`, so they do not identify individual C++ kernels. Julia sampling,
cProfile times, and Rprof percentages are not directly comparable. Per-input
hotspots and sample counts are retained in the profile index.

The complete timing/profiling comparison above is validated. The full
230-observation fresh-process memory rerun is **unfinished**. After the core
comparison, a bounded focused attempt (job 13424709) reached its wall limit
before its planned 60 observations completed. Its raw outputs are retained in
[focused-memory-raw.tar.gz](focused-memory-raw.tar.gz); they are not mixed with
the earlier CPU's memory summaries or presented as a complete five-replication
memory comparison. Completing the all-method memory pass requires the pending
extra-budget approval. The task used 119 min 46 s of the standing 120-minute
running-time allowance, with at most 8 CPUs and 16 GB. No further numerical
jobs are running or authorized beyond that ceiling yet.



Full package suites passed on Discovery: **2,925/2,925** checks on Julia 1.13.2
and **2,927/2,927** on Julia 1.10.12, including Aqua and the new optional-backend,
weighted FISTA, path/CV, Float32, iteration-limit, and diagnostic corner cases.
The public API canary also passed two medium and two wide independent inputs
at all three penalties. The first setup attempt spent its time on new-CPU
precompilation and exposed eight near-zero residual assertions with an overly
strict relative-only tolerance; correcting the assertions' absolute roundoff
allowance produced the complete passing suites. Production solver code did
not change between these tests and measurements.

The measured production implementation is commit
`267430fc240841e072aab77cf6341bd135a74076`. Before baselines use the frozen
prior public API/environment from the preceding campaign. Package versions,
Julia manifests, input hashes, CPU details, and measured source hashes are in
[provenance](provenance/), with exact staged harness versions in
[harness-sources.tar.gz](provenance/harness-sources.tar.gz). The Julia wrappers use the pinned comparison
adapters; Python and R numerical workers and inputs are unchanged.

The run is `20261025-cd81-solvers`, under
`/home1/mkolar/juliaPkg/runs/20261025-cd81-solvers`. The original preparation
job 13424142 timed out after compilation; 13424248 passed both complete
suites. Comparison job 13424245 reached its wall limit in the final wide
worker. A Python CLI argument error is retained in its logs. Missing-only
recovery job 13424632 used the corrected launcher, kept completed receipts
and flushed Julia measurements, and finished ordinary and R component work.
An adaptive-profile counter scope error was then repaired in job 13424684,
which passed all five profile replications and completed independent checking
and core delivery without rerunning completed timings. Original logs and partial outputs remain available remotely. The
cancelled pending recovery submission 13424602 consumed no running time.

See the [driver and recreation recipe](../../improvements/README.md),
[complete timing table](full-table.md), [all timings](all-timings.csv),
[paired ratios](paired-ratios.csv), [independent certificates](independent-certificates.csv),
[profile index](profile-index.csv), [profiles](profiles.tar.gz), and
the memory status above. The parent workspace keeps the
lightweight run record at `.slurm/runs/20261025-cd81-solvers.json`.
