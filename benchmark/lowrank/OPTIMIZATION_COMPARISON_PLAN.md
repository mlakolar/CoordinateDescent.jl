# Optimization comparison plan — issue 81

Status: completed on 2026-10-25 after explicit user approval; five independent Python/Julia/R replications on Discovery. MATLAB was excluded. See [measured results](optimization/results/discovery-20261025/README.md).

An initial plan was written before connecting to Discovery. Its unchanged copy is in `.slurm/runs/20261025-cd81-optimization-plan/initial-plan.md` in the parent workspace. This revision incorporates the license findings and the source review below. The text below records the approved scope; execution and subsequent numerical findings are recorded in the linked results.

## Scope and authorization

Compare optimization procedures for the implemented objectives: fixed-rank reduced-rank regression, coefficient nuclear-norm regression, row-sparse reduced-rank regression, and sparse singular-vector regression. Evaluate computational cost and numerical solution quality, with fixed data, ranks, penalties, weights, and initializations. Prediction, cross-validation, and statistical model comparisons are outside this campaign.

The user approved this revised plan and explicitly authorized five independent Python/Julia/R benchmark and profiling replications on Discovery. MATLAB is excluded. A MATLAB installation or visible license pool is not reported as a successful license checkout.

## Revised comparison matrix

| Objective | Core competitors | Algorithm question |
|---|---|---|
| Fixed-rank least squares | Our compact-SVD solver; R `rrpack::rrr` (QR/SVD) and `rrpack::rrr.fit` (normal equations); NumPy/SciPy compact-SVD adapter | Cost of closed-form factorizations and coefficient reconstruction. Add Julia projected proximal gradient with `IndBallRank(r)` on the medium and wide cases as an algorithm comparison against the known optimum. |
| Coefficient nuclear-norm regression | Our restarted accelerated proximal gradient; ProximalSolvers/ProximalOPT accelerated proximal gradient and ADMM; independent JuliaFirstOrder `FastForwardBackward` and `PANOC` with ProximalOperators; Python PyProximal FISTA and ADMM with a cached multiresponse least-squares proximal operator | Specialized versus generic proximal algorithms, acceleration/restart, splitting with reusable linear solves, and proximal quasi-Newton acceleration. Plain proximal gradient is a limited medium/ill-conditioned-case ablation. |
| Row-sparse, rank-constrained regression | Our block coordinate descent; R `rrpack::srrr`; a Julia adapter using proximal gradient/FISTA for the A block, with the same Procrustes V update | Whether full-block proximal steps outperform row coordinate descent within the same nonconvex factorization. |
| Sparse singular-vector regression | Our component solver; R `rrpack:::surr_fit_Rcpp`; a Julia adapter replacing its predictor lasso coordinate updates with proximal gradient/FISTA | Compare the same weighted rank-one component at fixed penalties, with identical normalization, starting factors, and exclusive response layers. |

The adapters above are planned benchmark code, not existing turnkey regression APIs. Their equations and final certificates must pass the pilot before any timing claims. PyProximal has the needed nuclear penalty and FISTA/ADMM interfaces; JuliaFirstOrder provides nuclear and rank-set operators plus first-order and PANOC solvers. The local ProximalSolvers source explicitly supports matrix-valued multiresponse least squares with a nuclear penalty, and its least-squares proximal operator caches a thin SVD. Its parent folder is currently named `ProximalOPT.jl`; record the actual module and revision used.

Use CVXPY with SCS as a small convex-reference experiment, not the primary speed target. Match our coefficient penalty `lambda * ||B||_*`; rrpack's adaptive fitted-value penalty is a different objective and is excluded. SOFAR, glmnet, PLS, and predictive model comparisons are outside scope. MATLAB iRRR/TFOCS are removed from the core campaign because usable MATLAB access was not verified.

Proximal feasibility is already clear from source: our nuclear solver uses singular-value thresholding, and the sparse methods use lasso/group proximal updates. The unresolved question is which implementation and algorithm is fastest at a specified accuracy. The proximal map of a row-group penalty plus a rank constraint is not obtained simply by composing soft thresholding and rank truncation; preserve the factorized A/V problem instead.

## Fairness and measurements

Use identical Float64 inputs, centering, objective scaling, penalties, adaptive weights, starting points, and single-threaded workers. Generate each input once and load the identical values in each language; equal seed integers across languages do not guarantee equal inputs. Do not form a Kronecker design in a newly written adapter. Verify shared objective values and independent optimality/stationarity residuals before comparing times. A common tolerance is an externally measured residual, not an equal numerical setting in unrelated packages. Nonconvex methods require matching initialization and comparison of attained objectives, not global-optimum claims. In wide designs compare fitted values and objectives; minimum-norm and pivoted-QR coefficient representatives need not agree.

For the convex nuclear problem, independently calculate a primal-dual gap and a proximal-gradient residual at the returned coefficient matrix. For centered X,Y, put R=Y-XB, scale Theta=R/n until `||X'Theta||_op <= lambda`, and use the feasible dual value `<Y,Theta> - n*||Theta||_F^2/2`. Normalize the gap by `max(1,abs(primal),abs(dual))`. Evaluate the proximal-gradient mapping at step `1/L`, with `L=||X||_op^2/n`, and normalize its Frobenius norm by `max(1,||X'Y/n||_F)`. The main accuracy target is at most 1e-6 for both certificates. Separate convergence traces record when 1e-4, 1e-6, and 1e-8 are reached. Keep setup and certificate-evaluation costs visible; a library's reported convergence flag is not a certificate.

For fixed-rank regression check rank feasibility and loss against the closed-form optimum. For row-sparse regression recompute the row-group KKT residual and the response-subspace stationarity residual at the final A,V. For sparse components audit the weighted component objective and both factor-block residuals, including zero factors and rescaling. Target normalized block residuals at 1e-6. Report attained objectives alongside runtime and distinguish different stationary points. Do not time information-criterion selection as if it were a fixed-penalty solve.

Measure end-to-end fit time, repeated warm fit time, setup/factorization cost, peak process RSS, objective and residual versus time, iteration counts, and convergence failures. Measure an eight-level nuclear-norm path separately from independent cold solves; give each competitor the same warm-start and caching opportunity. Public API timings include centering and required setup; a separate common-input kernel measurement identifies wrapper and factorization overhead. Keep JIT compilation/interpreter startup, input loading, diagnostic traces, and profiling overhead separate from steady-state timing. Use final certificate checks outside the primary fit timer and calibrate package stopping rules in the pilot. Convergence traces are a separate instrumented experiment. Report process RSS with its interpreter baseline and startup/warm-up inclusion stated; language-specific allocated-byte counts are supplementary only.

Use this six-case grid, with working rank r=3:

| Case | n | p | q | Design |
|---|---:|---:|---:|---|
| Small | 300 | 50 | 10 | Full rank, moderate conditioning |
| Medium | 300 | 50 | 100 | Full rank, moderate conditioning |
| Many responses | 300 | 50 | 1000 | Full rank, moderate conditioning |
| Wide | 200 | 400 | 100 | p > n |
| Ill-conditioned | 500 | 100 | 100 | Prescribed nonzero singular-value condition number 1e4 |
| Rank-deficient | 300 | 80 | 100 | Exact duplicated/linearly dependent columns, design rank 40 |

Use five independent input replications per case and three timing repetitions per replication. Use three fixed positive penalty strengths (0.5, 0.1, 0.02 of the shared objective's lambda-max); compute group/component lambda-max once with shared starting factors and weights. Primary sparse comparisons use unit weights; add a small adaptive-weight check on the medium case. Shared data may have low-rank structure to exercise the solvers, but no predictive performance will be measured.

Profile all five input replications at the middle penalty on the medium, many-response, and wide cases, separately from timing. Use Julia sampling profiles, Python profiling that exposes native SVD/BLAS time where available, and R profiling with native-call limitations stated. Retain raw reports and sample counts; do not compare language profiler percentages as if they were identical measurements. Rotate solver/language order within replications. Report median paired runtime ratios and their replication ranges, with every timeout/nonconvergence retained. Three repetitions on one dataset are timing repeats, not three independent problem replications.

Source review found that rrpack's active C++ sparse-component routine still forms an n*q by q Kronecker matrix for the response update/residual (`src/surr_fit.cpp`, lines 319 and 352), although its lasso solve uses sufficient statistics. At n=300,q=1000 this matrix alone contains 300 million doubles (2.4 GB decimal). This is a memory-risk prediction from source, not a measured benchmark result. Run the largest sparse-component case in an isolated step with up to 8 GB and retain any OOM/timeout as an outcome; do not launch four such workers concurrently.

## Execution after approval

1. Freeze source versions, package versions, shared inputs, objectives, and certificates. Start from branch commit `9b7edcf315f173e862b049519a9731d3c8f02925`, with original dependencies ProximalBase `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a` and ProximalSolvers `9041e236f3de8ffdc02103e5a505993463e00bb1`. Do not silently take unrelated dirty parent checkouts. Pin Python/JuliaFirstOrder/R versions when constructing the run environment. Keep all Julia preparation, package installation/precompilation, and numerical checks inside Slurm.
2. Run a correctness/cost pilot on two independent medium/wide inputs, an orthonormal nuclear reference, and a small CVXPY reference. Give the pilot at most 15 running minutes, including environment preparation. Verify equations and certificates, then set bounded iteration/time limits. If an adapter is unsupported or does not solve the same objective, record that and omit it; do not replace it with a different statistical estimator.
3. Run repeated measurements and profiling on Discovery `main`, account `mkolar_1314`, CPU constraint `epyc-7513`. Use GNU Parallel launching exclusive `srun --exclusive --exact -N1 -n1 -c1` workers, normally up to four 1-CPU/2-GB workers. Run the largest RSSVD memory case separately with one 1-CPU/8-GB step. Set Julia, BLAS, OpenMP, MKL, and OpenBLAS threads to one. Confirm runtime thread counts and record BLAS vendors; library differences are part of implementation results, not evidence about language alone.
4. Pilot and main measurements together have at most 60 running-union minutes; reserve up to 20 additional minutes only for bounded recovery. Start with a 30-second per-solve timeout and preserve partial convergence traces. Reduce optional ablations/reference coverage before reducing the five-replication core. If a fair core comparison cannot fit the ceiling, stop with its measured cost and request a revised scope/budget.
5. Report paired runtime ratios at matched solution accuracy, memory, convergence reliability, and bottlenecks. Separate algorithm comparisons, implementation comparisons, and packaging overhead. Retain raw tables, profiles, solver adapters, source/environment references, and a compact recreation recipe. Update the PR with evidence only after the measurements exist.

Proposed additional running-time ceiling: 60 minutes plus at most 20 minutes of bounded recovery, with at most 4 CPUs and 12 GB RAM concurrently. The existing task has used 30.4 running-union minutes and 1.976 allocated CPU-hours; task-wide limits remain 8 CPUs, 16 GB RAM, 120 running-union minutes, and 16 CPU-hours. Maximum proposed combined running time is therefore 110.4 minutes. No standing allowance overrides the user's explicit approval gate for this comparison.

## Discovery MATLAB check and resulting plan change

The installed modules are `matlab/2024a` and `matlab/2025b`; R2025b is launched through CARC's Apptainer wrapper. Both installations have a network license configuration, and their configured license-server TCP endpoint was reachable from Discovery. Slurm's `Licenses=(null)` is scheduler configuration and is not evidence about MATLAB entitlement.

License-only startup probes were bounded and ran without scientific calculations or Slurm submissions, under CARC's exception for small login-node tests. R2024a timed out at 60 seconds without producing a license result. R2025b terminated abnormally in runtime startup and reached the timeout/kill bound, without producing a license result. A headless R2024a probe reported MathWorks service-process launch failures (`Resource temporarily unavailable`). The server probe also encountered transient fork failures before succeeding. These are runtime/process-resource failures, not a definitive license denial; no successful `license('checkout','MATLAB')` result was obtained.

Updated decision: use Python, Julia, and R for the core comparison. Treat MATLAB availability for this account as **unverified**, not absent. iRRR/TFOCS require a later successful license checkout in a suitable environment and a toolbox-dependency audit before adding them; the core campaign does not depend on MATLAB. No extra toolbox access is assumed. No license credentials, server settings, or existing jobs were changed. All probes have ended, the detached service created by the headless probe was stopped, and this task's SSH connections are closed.

## Sources

- [rrpack reference](https://www.stat.ethz.ch/CRAN/web/packages/rrpack/refman/rrpack.html) and [active C++ sparse-component routine](https://github.com/cran/rrpack/blob/master/src/surr_fit.cpp).
- [JuliaFirstOrder algorithms](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable/guide/implemented_algorithms/) and [ProximalOperators functions](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/).
- [PyProximal nuclear operator](https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.Nuclear.html), [FISTA/proximal-gradient interface](https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.optimization.primal.ProximalGradient.html), and [ADMM interface](https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.optimization.primal.ADMML2.html).
- [CVXPY matrix norms](https://www.cvxpy.org/tutorial/functions/index.html).
- [CARC MATLAB policy and usage](https://www.carc.usc.edu/user-guides/research-applications/computational-mathematics/matlab) and [MathWorks license-test versus checkout semantics](https://www.mathworks.com/help/matlab/ref/license.html).

Execution record: .slurm/runs/20261025-cd81-optcomp.json in the parent workspace. Preparation, pilot, and calibration remain separate from the main measurements; source snapshots and failed pilot logs are retained. Main nuclear tolerances were calibrated toward a 1e-8 proximal residual so that the independently checked duality gap meets 1e-6. Sparse certificates retain actual factors; when the existing API returns only B, its frozen outer loop is replayed outside the fit timer to recover and verify the factors.

Completion: the campaign retained 5,460 timing observations, 290 profile reports, and 170 fresh-process memory observations. The 1,400 independent coefficient checks passed except for five normal-equation rrpack fits on ill-conditioned inputs; the QR route passed. A coefficient-only fixed-rank supplement separates public result diagnostics, and an explicitly oracle-assisted stopping ablation avoids attributing projected-gradient native stopping stalls to time required for the common loss target. Additional running-union time was 34.13 minutes and allocated CPU time was 1.969 CPU-hours, within the approved budget. No MATLAB measurements were run.
