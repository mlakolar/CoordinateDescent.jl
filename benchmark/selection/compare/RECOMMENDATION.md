# Selection implementation: keep, optimize, or replace

**Keep the native least-squares solvers, fixed-X construction, Gaussian sampler and CPSS wrapper. Prioritize statistical calibration and knockoff construction choices, then covariance-factor caching.** The direct measurements support retaining the core implementation. They do not support advertising the current defaults as a broadly competitive FDR procedure for arbitrary high-dimensional designs.

**Sparse follow-up:** this document preserves the original baseline study. Native sparse handling is now implemented and passes the accuracy, storage and performance targets in a separate seven-replication Discovery campaign. See the [follow-up report](../sparse/REPORT.md) for the current sparse recommendation. The statistical findings below remain relevant; the historical sparse runtime limitation is superseded.

All numerical work ran on Discovery. The primary study used seven independent process replications, three timing batches per case, 147 separately collected profiles, and 840 shared datasets (140 per scenario). Julia 1.10.12, SLOPE.jl 1.3.1, Knockoffs.jl 3.1.4 and GLMNet.jl 0.7.4 were compared with R 4.5.3, SLOPE 2.1.1, knockoff 0.3.6, glmnet 5.1 and stabs 0.7-1. The runtime source is commit `ab544aec2a4f24b41ec5befd4467024ce2e29f16`.

The [protocol](README.md), [primary results](discovery-20261025/results/summary.md), [validation receipt](discovery-20261025/results/validation.json), [raw selections](discovery-20261025/results/mc-julia-1.csv), and retained scripts provide the recreation recipe. Results are development evidence for these workloads, not a broad benchmark of all package features or independent paper confirmation.

## Runtime competitiveness

These are medians of within-process medians, in milliseconds. Fits use one BLAS/OpenMP thread. The explicitly labelled CPSS variant uses two Julia threads. Inputs, objectives, penalties, intercept/standardization conventions and path lengths are matched. Each SLOPE path has 20 levels from the null-model threshold down to 20% of that threshold. Long paths at much weaker regularization were not benchmarked. Startup, compilation, input/output and profiling are excluded. Allocations record only the Julia heap; foreign C++/Fortran allocations and peak resident memory are not comparable from these records.

| SLOPE path workload | Native CD | SLOPE.jl | R SLOPE auto |
|---|---:|---:|---:|
| Dense: n=600, p=200, rho=.3 | 4.589 | 8.159 | 17.000 |
| Wide: n=200, p=1000, rho=.3 | 6.815 | 15.095 | 22.000 |
| Correlated: n=p=400, rho=.8 | 4.982 | 10.281 | 17.000 |
| Sparse: n=600, p=1000, density=.02 | 13.627 | 8.383 | 19.500 |

Paired comparisons in this baseline study show the native solver is **1.77–2.13 times faster than SLOPE.jl on the three dense cases**, and 1.47–3.51 times faster than R auto across all four. On sparse input it is **1.59 times slower than SLOPE.jl**, with a replication range of 1.41–1.83. That baseline revision densified sparse matrices. The [native sparse follow-up](../sparse/REPORT.md) resolves this bottleneck on the tested workloads. A full replacement of the dense solver is not supported by these results. The existing packages still provide broader GLM and tuning functionality ([SLOPE features](https://jolars.github.io/SLOPE/)).

| Component workload | Native CD | Julia alternative | R alternative |
|---|---:|---:|---:|
| Fixed-X construction, n=600,p=200 | 15.094 | 40.348 (Knockoffs.jl) | 55.000 (knockoff) |
| Gaussian equicorrelated construction, n=600,p=200 | 12.267 | 5.859* | 32.000 (knockoff) |
| Gaussian equicorrelated construction, n=200,p=1000 | 369.305 | 158.908* | 1945.000 (knockoff) |
| Shared augmented-design lasso, n=600,2p=400 | 1.545 | 27.585 (GLMNet.jl) | 8.000 (glmnet) |
| CPSS, 30 pairs, fixed-lambda capped base | 22.527 serial; 15.995 threaded | 143.508 (matched GLMNet.jl loop) | 122.000 (matched serial stabs) |

*Knockoffs.jl's native Gaussian timings are retained for transparency, but its tested sampler fails the covariance check below. They are not evidence of a faster valid construction.*

The native fixed-X constructor is 2.66 times faster than Knockoffs.jl and 3.38 times faster than R by paired medians. Native serial CPSS is 5.52 times faster than the matched serial stabs wrapper. Its two-thread paired median speedup is 1.46, range 1.27–1.71. These comparisons use an identical fixed-penalty capped selector; they do not rank each package's different default cross-validation or parallel workflow.

The common optimality certificate has maximum relative proximal residual **7.37e-8** across all benchmark fits. All **9,800 matched primary method–dataset selection comparisons agree exactly**. CPSS benchmark frequencies also agree exactly. Fixed-X maximum Gram and cross-Gram discrepancies are 1.83e-13 and 2.99e-8. The latter includes the R construction's different PSD safety adjustment. An optional R FISTA pilot had insufficient accuracy at tol=1e-8; a much tighter calibration became costly and was stopped. Its uncalibrated timings are not included in these rankings.

See [paired runtime ratios](discovery-20261025/results/paired-time-ratios.csv), [accuracy summaries](discovery-20261025/results/accuracy-summary.csv), and [selection agreement](discovery-20261025/results/selection-agreement.csv).

## Statistical performance changes the priorities

The shared Gaussian designs use population covariance, known noise standard deviation one and 20 contiguous alternating-sign signals. The standard signal magnitude is 4.5/sqrt(n), reduced to 3/sqrt(n) in the weak case. Most scenarios have n=300,p=100; the wide scenario has n=180,p=300. Alternating signs deliberately stress correlated-feature selection. The dramatic differences below should not be generalized to every signal pattern.

### SLOPE: retain the optimizer; change the calibration contract

The nominal q is .1. Values below are estimated FDR / power, in percent. All backends agree at identical penalty weights.

| Design | Plain BH weights | Gaussian-adjusted weights |
|---|---:|---:|
| iid Gaussian | 18.46 / 95.39 | 7.08 / 86.86 |
| AR(.4) | 25.52 / 32.82 | 20.70 / 21.96 |
| AR(.8) | 16.46 / 4.25 | 14.87 / 3.71 |
| Weak AR(.4) | 17.16 / 8.11 | 14.73 / 6.57 |
| Global null AR(.4) | 7.14 / — | 7.14 / — |
| Wide AR(.4) | 62.88 / 24.71 | 51.95 / 12.61 |

The BH FDR Monte Carlo SE is .0062 for iid and .0132 for the wide case; Gaussian-adjusted SEs are .0050 and .0218. The iid improvement is substantial. The adjusted sequence still fails to deliver FDR .1 in the correlated AR(.4) and wide designs. The AR(.8) estimate is less precise (SE .0259), so its point estimate alone is weaker evidence of excess FDR.

The Gaussian-adjusted follow-up reused the same saved datasets after the primary result exposed inflation. It tests an existing option, rather than tuning weights individually to these outcomes. CD, SLOPE.jl and R agree on all 1,680 additional comparisons with common weights. R's native Gaussian-weight option differs slightly from SLOPE.jl's sequence (maximum penalty difference 9.62e-5), producing four different selection sets among 840 datasets; conclusions are unchanged.

**Recommendation:** keep BH weights as an explicitly orthogonal-design option. Add Gaussian-adjusted weights with clear assumptions and empirical limitations, and avoid presenting a general-design `q=.1` call as achieving FDR .1. Replacing the optimizer with SLOPE.jl or R does not repair this issue. General correlated-design error control needs a different justified calibration or an appropriate knockoff procedure.

See [Gaussian-weight results](discovery-20261025/results/gaussian-simulation-summary.csv) and [follow-up validation](discovery-20261025/results/gaussian-validation.json).

### Knockoffs: retain the correct sampler; add MVR/max-entropy choices

All valid shared-sampler comparisons use exactly the same X, y, Gaussian innovations, lasso penalty .04, solver and knockoff+ target .1. Only the diagonal construction changes. MVR and maximum-entropy diagonals come from Knockoffs.jl and are used with our validated conditional sampler.

| Design | Equicorrelated power | MVR power | Maximum-entropy power | MVR estimated FDR |
|---|---:|---:|---:|---:|
| iid | 88.71% | 88.71% | 88.71% | 9.05% |
| AR(.4) | 0.29% | 44.43% | 41.46% | 7.24% |
| AR(.8) | 0.14% | 0.36% | 0.14% | 0.75% |
| Weak AR(.4) | 0.00% | 4.50% | 3.29% | 2.34% |
| Wide AR(.4) | 0.00% | 1.25% | 0.39% | 2.00% |

For AR(.4), the paired MVR power gain is **44.14 percentage points, MC SE 2.82 points**. Fixed-X knockoff+ also has useful power there: 45.11%, estimated FDR 5.74%. Fixed-X is unavailable in the wide case. None of the tested valid knockoff settings has useful power in the severe-correlation or wide scenarios. This is a limitation of the tested fixed-lambda procedure; package-default cross-validated statistics and grouped knockoffs were not evaluated.

**Recommendation:** replace the equicorrelated-only policy with explicit MVR and maximum-entropy options, initially by reusing diagonal solvers through the existing supplied-s interface. Preserve the native sampler, threshold and lasso engine. Evaluate swap-equivariant tuning or other valid model-X statistics next, especially for wide designs. Fixed-X tuning must retain its separate sufficiency requirements.

#### Defect in the tested Knockoffs.jl sampler

For conditional covariance C, the tested source obtains `L = cholesky(..., C).L` and draws `Z * L`. Rows of that product have covariance L' L, while the required covariance is L L' = C. The multiplication needs the corresponding transpose/upper factor.

Seven replications with 20,000 conditional draws each, at X=0,p=30,AR(.8), give median empirical relative covariance error .2538 for Knockoffs.jl, versus .0323 for CD and .0321 for R. Recovering the sampler's linear map from shared normal innovations gives implied covariance error **.25080186** for Knockoffs.jl, versus at most 1.12e-15 for CD and 9.99e-16 for R. This isolates a matrix-factor orientation defect rather than Monte Carlo fluctuation.

Keep native Knockoffs.jl sampler outputs out of valid FDR/power rankings until corrected. Its diagonal optimizers remain useful: the shared-sampler comparisons check their feasibility and avoid this defect. The tested upstream source was unchanged.

See the [installed source](discovery-20261025/metadata/Knockoffs-modelX-installed.jl#L110) and [constructor diagnostics](discovery-20261025/results/constructor-validity.csv).

### CPSS: retain the fast wrapper; make the error-count tradeoff explicit

| Design | Cap 10: power / mean false selections | Cap 20: power / mean false selections | R path selector, cap 10: power |
|---|---:|---:|---:|
| iid | 15.14% / 0.000 | 58.25% / 0.014 | 10.46% |
| AR(.4) | 6.89% / 0.014 | 23.14% / 0.107 | 3.68% |
| AR(.8) | 1.93% / 0.036 | 4.43% / 0.300 | 1.57% |
| Weak AR(.4) | 3.18% / 0.014 | 9.29% / 0.171 | 2.18% |
| Wide AR(.4) | 0.43% / 0.007 | 1.11% / 0.007 | 0.82% |

At p=100 and cutoff=.8, the basic bounds are 1.667 for cap 10 and 6.667 for cap 20, conditional on the documented null-selection-probability requirement. Raising the cap improves power but quadruples that error-count bound. Under the global null, cap 20 selects .586 false variables on average and has estimated FDR .436; cap 10 selects .079 on average and has FDR .079. Thus a larger cap is not a way to obtain FDR .1. These are distinct error criteria.

**Recommendation:** keep the deterministic, reproducible wrapper and capped-lasso selector. Add a PFER-oriented parameter interface and optional base-selector/path support. The R path selector was not a general power improvement on these cases, so replacing the current selector wholesale is not justified. Report CPSS as an error-count procedure with its assumptions, rather than as interchangeable with FDR .1 methods.

## Engineering priorities

1. Add MVR/max-entropy diagonal choices and clarify SLOPE calibration before promoting broad FDR functionality. These changes address much larger practical gaps than changing optimizer backends.
2. Cache covariance-dependent Gaussian sampling factors and try Cholesky before a full conditional-covariance eigendecomposition, retaining the eigen fallback for PSD roundoff. LAPACK's eigensolver appears among the top sampled active leaves in all seven p=1000 constructor profiles; solves/factorizations also recur. The cache optimization has not yet been implemented or benchmarked.
3. Native sparse SLOPE handling is completed in the [follow-up](../sparse/REPORT.md). The old profiles' dense matrix-vector products and array copies motivated that change; use the new replicated profiles before further optimization.
4. Improve statistical tuning and extend the tested scope before claiming strong wide-design selection performance. Include valid model-X tuning/statistics, covariance estimation/misspecification and additional signal patterns in that next study. Keep fixed-X and CPSS guarantees separate.

Keep PR #84 draft while deciding the calibration and construction interface. The benchmark results support keeping its computational core; they also identify the statistical limitations that the next implementation changes should address. This assessment does not implement those proposed changes.

## Retained run

Run ID: `20261025-cd82-compare`. Production job: `13423560`; profile aggregation: `13423579`; Gaussian-weight follow-up: `13423674`. Preparation and pilot outcomes, including the stopped optional FISTA calibration, are retained in the project registry and metadata. Remote root: `/home1/mkolar/juliaPkg/runs/20261025-cd82-compare`.

The [full primary table](discovery-20261025/results/simulation-summary.csv), [paired construction-choice power differences](discovery-20261025/results/paired-power-differences.csv), [profile summaries](discovery-20261025/results/profile-summary.csv), environment manifests, raw per-shard selections, logs and accounting are retained alongside this report. Generated matrices remain on Discovery and can be recreated from the saved seeds.

Including preparation, pilots, the stopped FISTA calibration, all earlier issue-82 validation jobs and this comparison, the whole task used **66.83 minutes of allocation-running union time and 3.706 allocated CPU-hours**, with peak allocated resources of six CPUs and 16 GB. It remained within the standing eight-CPU/16-GB/two-hour allowance. See [accounting](discovery-20261025/metadata/accounting.tsv) and [resource summary](discovery-20261025/metadata/resource-summary.json).
