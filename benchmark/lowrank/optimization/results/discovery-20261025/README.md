# Python, Julia, and R optimization comparison

A subsequent [solver-option follow-up](../discovery-solvers-20261025/README.md)
implements PANOC, block FISTA, and diagnostic reuse. Its measurements use
a different fixed CPU model and include fresh before/after baselines.

The implementation is competitive, with clear opportunities to improve it.
Its nuclear solver beats the tested Python FISTA/ADMM adapters, but PANOC
often wins within Julia. Row-sparse coordinate descent beats rrpack at the
middle penalty, while block FISTA improves several designs. The sparse
component solver is particularly effective against rrpack's implementation.
Fixed-rank coefficient computation is competitive with SciPy; the public
Julia API additionally computes result diagnostics that materially increase
its fit time. These are results for the tested inputs and implementations,
not a claim of universal superiority.

All numerical work, environment preparation, timings, profiles, and numerical
aggregation ran in Discovery Slurm allocations on AMD EPYC 7513 CPUs. There
were five independent shared inputs per design and three timing rounds per
input. Tables show milliseconds: the median of the five within-input timing
medians. Each worker used one CPU and one BLAS/OpenMP thread. MATLAB was
excluded. Prediction and statistical performance were not evaluated.

The six cases are small (300,50,10), medium (300,50,100), many responses
(300,50,1000), wide (200,400,100), ill-conditioned (500,100,100; condition
number 1e4), and deficient (300,80,100; design rank 40). Tuples are (n,p,q).
Working rank is 3. The tables below use a fixed penalty of 0.1 times the
shared objective-specific lambda-max, or fixed rank 3.

For nuclear regression the penalty is on **B**, not XB:
`||Yc-Xc B||_F^2/(2n) + lambda ||B||_*`. Its proximal operator is singular-value
soft thresholding. All seven proximal implementations solve this same
convex objective through the supplied adapters.

| Case | Our APG | Julia PANOC | ProximalSolvers APG | Python FISTA | Python ADMM |
|---|---:|---:|---:|---:|---:|
| Small | 2.04 | 1.63 | 2.49 | 7.60 | 8.11 |
| Medium | 25.79 | 16.94 | 28.41 | 52.68 | 49.45 |
| Many responses | 146.87 | 151.04 | 224.33 | 304.58 | 290.68 |
| Wide | 386.64 | 240.73 | 314.22 | 848.92 | 804.53 |
| Ill-conditioned | 154.16 | 96.25 | 153.43 | 441.03 | 397.44 |
| Deficient | 32.93 | 22.44 | 36.58 | 60.25 | 65.03 |

At this penalty, median paired alternative/our runtime ratios are 1.82–3.71
for Python FISTA and 1.92–3.97 for Python ADMM. PANOC's ratios are 0.57–0.80
on five designs, with substantial replication variability in some cases;
the many-response ratio is 1.05. ProximalSolvers APG also wins on the wide
case (ratio 0.82) and is competitive on the ill-conditioned case.
JuliaFirstOrder FISTA and ProximalSolvers ADMM were also measured; see the
[complete table](full-table.md).

Warm paths change the comparison. For eight descending penalties, PANOC/our
paired ratios are 0.89, 1.06, 1.18, 0.57, 0.58, and 1.07 for small, medium,
many-response, wide, ill-conditioned, and deficient cases respectively.
All solvers receive caching and warm starts. The tables retain final path
endpoint certificates; independent coefficient-file checks cover cold fits,
not every intermediate path point.

Row-sparse regression uses the same A/V factorization, shared initial factors,
unit row weights, and objective scaling. Block FISTA replaces the predictor
coordinate updates and retains the Procrustes response update.

| Case | Our coordinate descent | Block FISTA | R rrpack srrr |
|---|---:|---:|---:|
| Small | 0.54 | 0.59 | 6.17 |
| Medium | 1.41 | 0.85 | 7.61 |
| Many responses | 8.74 | 3.29 | 34.64 |
| Wide | 22.06 | 34.70 | 30.31 |
| Ill-conditioned | 41.25 | 16.43 | 164.98 |
| Deficient | 3.78 | 1.48 | 19.88 |

At this penalty, block FISTA/our paired ratios are 0.39–0.59 on medium,
many-response, ill-conditioned, and deficient designs; coordinate descent
wins on small and wide designs. R/our paired ratios range from 1.33 to 11.27.
Attained objectives agree to numerical precision at the middle penalty.
Across all three penalties, block FISTA's maximum relative objective
difference is below 5.2e-16.

There is a consequential nonconvex exception: on the wide design at penalty
fraction 0.5, rrpack is faster but reaches a worse stationary point on four
of five inputs, with a relative objective difference up to 0.163. Its
median runtime ratio 0.52 is therefore not an accuracy-equivalent speed win.
For example, the first input has objectives 641.02 (ours) and 713.09 (R).
Stationarity is a local certificate, not global optimality.

A separate medium-case adaptive-weight experiment used five shared inputs
and inverse-row-norm-squared weights. All 30 saved fits passed factor
certificates, with maximum relative objective difference below 2.7e-15.
At the middle penalty Julia took 1.83 ms and R 3.40 ms; the paired ratio was
1.86. This supplement supplies centered inputs before its timer, so its
times should not be equated with the full-fit table above.

Sparse-component timings compare one exclusive rank-one component at fixed
penalties, not a full information-criterion or rank-selection path.

| Case | Our coordinate descent | Component FISTA | R rrpack component |
|---|---:|---:|---:|
| Small | 0.18 | 0.47 | 2.46 |
| Medium | 0.42 | 0.81 | 14.85 |
| Many responses | 2.95 | 3.66 | 1142.46 |
| Wide | 5.59 | 43.48 | 20.55 |
| Ill-conditioned | 2.18 | 7.85 | 36.76 |
| Deficient | 0.66 | 1.34 | 17.61 |

Our component solver wins on all six middle-penalty cases. R/our median
paired ratios range from 3.74 to 342.80, with attained objectives agreeing
within 5.5e-14 relatively across the full component grid. The many-response
advantage partly reflects rrpack materializing an n*q by q Kronecker matrix:
300 million doubles, or 2.4 GB decimal, before other workspace. This is an
implementation difference, not an inherent lower bound for R algorithms.

Fixed-rank regression has a closed-form solution, so iterative projection
is possible but unnecessary for these dense problems.

| Case | Our public fit | Our coefficient computation | SciPy coefficients | R QR/SVD fit |
|---|---:|---:|---:|---:|
| Small | 1.29 | 0.77 | 1.02 | 3.49 |
| Medium | 2.37 | 1.34 | 1.63 | 9.97 |
| Many responses | 8.42 | 4.20 | 4.60 | 113.67 |
| Wide | 20.48 | 11.91 | 13.77 | 31.24 |
| Ill-conditioned | 7.75 | 4.43 | 5.12 | 20.10 |
| Deficient | 3.33 | 1.72 | 2.05 | 10.75 |

The public Julia result includes rank, loss, and degrees-of-freedom
diagnostics. The coefficient-only supplement excludes those diagnostics and
retains centering and factorization, matching the work returned by the
SciPy adapter. It ran in a separate allocation on the same CPU model;
small timing differences should not be interpreted as precise paired speedups.

R's separate `rrr.fit` routine forms normal equations and uses a default
pseudoinverse cutoff that drops valid directions in the ill-conditioned
case. All five such inputs fail the loss-accuracy target (relative errors
8.8e-5–6.0e-4). These adverse outcomes remain in every relevant table.
The added `rrpack::rrr` QR/SVD route, with an explicit rank tolerance, passes
all 30 inputs and is the accuracy-qualified R comparator above.

JuliaFirstOrder rank projection also solves the problem. Its original
native-stopping experiment had three 30-second capped rounds despite
meeting the external loss target. An explicitly favorable, oracle-assisted
ablation stops once relative loss differs from the *known optimal loss*
by at most 1e-8. It takes 14.02 ms on medium and 525.33 ms on wide, versus
2.37 and 20.48 ms for the public closed-form fit. All five inputs pass.
This oracle stopping rule is a benchmark aid, not a deployable solver rule;
the original capped observations are retained in `all-timings.csv`.

The profiles support three concrete follow-ups: reduce repeated result
diagnostic SVDs in fixed-rank fits; consider a PANOC option for nuclear fits;
and consider block FISTA for selected row-sparse regimes. No production
solver was changed based on this campaign.

All five many-response profiles put ProximalBase's proximal SVD at the top
of our nuclear solver's self-sample frames. Python FISTA similarly spends
most native time at SVD/spectral call boundaries. PANOC profiles still
show SVD and allocation costs. Sparse coordinate methods show meaningful
allocation/indexing overhead, with repeated SVDs visible for row-sparse
fits. R component profiles attribute nearly all time to `.Call`; Rprof
does not resolve the C++ loop or attribute that time directly to the
Kronecker allocation. The source and measured RSS supply the latter evidence.
Profiler percentages across Julia sampling, cProfile, and Rprof are not
interchangeable. Requested R sampling intervals can be coarsened by the
runtime; retained reports state the actual interval.

Fresh-process RSS measurements include interpreter, imports, compilation,
warm-up, and the solve. On the many-response case, median peak/baseline
MiB are 864.8/789.1 for our nuclear solver, 139.2/116.4 for Python FISTA,
765.8/747.6 for our component solver, and 2558.2/263.9 for the R component.
The Julia harness imports all Julia competitor packages, so this is its
process footprint, not the minimal footprint of CoordinateDescent alone.
High-water baselines cannot be subtracted to recover allocated bytes.
See [all memory observations](memory.csv) and [replication summaries](memory-aggregate.csv).

The campaign retained 5,460 timing rows (5,250 core, 90 adaptive, 90
coefficient-only, 30 oracle-stopping), 290 separate profile reports,
170 fresh-process memory observations, and 1,400 independent saved-fit
checks. Of these checks, 1,395 pass; the only failures are the five default
normal-equation R fits described above. Convex nuclear checks require both
relative primal-dual gap and normalized proximal-gradient residual <=1e-6.
Fixed-rank checks require feasible rank and reference loss; sparse checks
use actual factors, row KKT residuals, response-subspace stationarity, and
orthogonality. The independent CVXPY/SCS orthonormal reference agrees with
the closed form to 1.52e-11 in maximum coefficient error.

The measured code is package commit `9b7edcf315f173e862b049519a9731d3c8f02925`,
with Julia 1.13.2, ProximalAlgorithms 0.7.0, ProximalOperators 0.16.1,
ProximalBase 0.5.3, ProximalSolvers 0.1.1, NumPy 2.4.6, SciPy 1.17.1,
PyProximal 0.13.0, R 4.5.2, and rrpack 0.1-14. BLAS builds differ across
environments, so these measurements compare implementations, not languages
in isolation. Exact versions, revisions, and worker source hashes are retained.

The run is `20261025-cd81-optcomp`, rooted at
`/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp` on Discovery. Preparation,
pilots, recoveries, measurements, and aggregation used 34.13 running-union
minutes and 1.969 allocated CPU-hours, within the approved 60-minute ceiling.
The [accounting record](provenance/accounting.psv) includes every attempt.
Initial interface/serialization mistakes and a GNU Parallel environment
variable conflict were corrected before final coverage; superseded outputs
and the original capped/failed outcomes remain in the run directory.

Reporting files:

- [Full middle-penalty table and ranges](full-table.md), [all core observations](all-timings.csv),
  [replication medians](replication-medians.csv), [paired ratios and objective differences](paired-ratios.csv).
- [Core independent certificates](independent-certificates.csv),
  [adaptive results](adaptive/all-timings.csv), [coefficient-only results](rank-core/aggregate.csv),
  [oracle-stopping results](rank-matched/aggregate.csv), and [final coverage](final-summary.json).
- [Profile index with hotspots and sample counts](profile-index.csv) and
  `profiles.tar.gz`: 290 readable reports plus 110 native profiler files.
- `diagnostics.tar.gz`: all 30 input metadata files, per-worker Python/R
  environment records, setup observations, and instrumented convergence traces.
  Use `tar -tzf` to list either archive and `tar -xzf` to inspect it.
- [Julia manifest](JuliaManifest.toml), [Julia project](provenance/JuliaProject.toml),
  [Python versions](provenance/python-freeze.txt), [R thread/BLAS check](provenance/r-configuration.txt),
  and [measured worker hashes](provenance/measured-source-sha256.txt).

Raw shared matrices and saved coefficients remain in the run directory;
they are recreated from the retained generator, seeds, and exact environments.
See the [harness and recreation instructions](../../README.md).
