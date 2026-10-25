# Issue78 solver study

This is an experimental benchmark and correctness suite for [issue78](https://github.com/mlakolar/CoordinateDescent.jl/issues/78). It does not yet add a supported quantile-regression API, pivotal penalty selection, CV, post-selection refits, or debiased inference.

See [QR_AUDIT.md](QR_AUDIT.md) for the historical solver choice and parametric-simplex considerations.

## Matched objectives and diagnostics

All methods fit an unpenalized intercept and the objective
`mean(rho_tau(y-a-X*b)) + lambda*sum(abs,b)`. The timing designs have columns normalized to RMS1, and use unit loadings for every method. The golden design is centered and normalized to sample SD1 to match conquer's internal standardization. `quantreg::rq.fit.lasso` minimizes a sum of losses, with a half-weighted absolute penalty from its pseudo-observations, so it receives `2*n*lambda`; rqPen receives `scalex=FALSE`, `alg="br"`, and the original lambda. Conquer receives the parabolic (Epanechnikov) kernel and the same bandwidth as Julia. Its default Gaussian kernel is a different smoothing objective.

The exact dual is `max y'u/n`, with `sum(u)=0`, `tau-1<=u<=tau`, and `abs(X'u)/n<=lambda`. Certificates project the supplied dual onto the intercept/box constraints, then contract it to enforce the penalty constraints. Every returned solution retains its exact primal objective and a feasible-dual lower bound. Exact solves qualify at gap `<=1e-5*(1+abs(primal))`; negative gaps below `-1e-7` fail.

Convolution smoothing uses the existing `ProximalBase.ConvolutionCheck` and the production proximal Newton `GLMLassoPath`, plus an experimental accelerated proximal gradient implementation. A smoothed fit qualifies by a proximal KKT residual `<=2e-5`; its exact-objective gap is separately reported. At bandwidth h the Epanechnikov smoothing lies between the check loss and check loss plus `3h/16`, but that bound does not establish estimator agreement or inferential equivalence in finite samples. The ADMM candidate is a linearized two-block implementation with a power-iteration majorizer; it is one prototype, not a comparison against every ADMM variant.

## Measurement design

All fitting, tests, timing, profiling and numerical summaries run under Slurm on Discovery. Julia/BLAS/OpenMP and LP solvers use one thread. Independent workers run through GNU Parallel and exclusive srun steps, bound to allocated cores. Other workers and cluster jobs may share memory bandwidth; node identity is retained.

A worker compiles every method on smaller data of the same dense/sparse type, generates one input, then runs two independently shuffled blocks of full eight-level paths. Three seeds per stratum give independent input/process replicates; the two within-process measurements are not counted as independent inputs. Summaries use arithmetic means of per-input means, input SD, and a Student-t interval with input as the unit. The small input count and shared-node setting limit precision. Minimum times are not the decision statistic.

Measured wall time includes model construction, objective/bound updates, solver work and solution/dual extraction. A fresh LP is built for every lambda in a cold path; a warm path starts with a new model and retains it within that path. No state leaks between measured paths. LP setup, optimize and extraction times are also retained. Input generation, compilation, explicit premeasurement GC, and the external certificate pass are outside the timed region. Julia allocation/GC counts exclude most native LP solver allocations and cannot be read as total solver memory.

The LP time budget covers an entire path, not each lambda. Partial paths, solver timeouts, errors and failed certificates remain visible and are excluded from speed claims. The production Newton routine has iteration limits but no strict wall-time callback; the outer worker has a900second cap. Profiling is separate from timed observations and includes native frames where Julia's sampling profiler exposes them. Native simplex pivots and phase timings supplement those traces.

## Recreate

The run registry is `.slurm/runs/20261025-cd78-quantile.json` in the workspace containing the checkout. It records exact source revisions, job IDs, resources and the staged harness versions. Remote run directory: `/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile`. The resolved Julia Manifest and R session information are retained with results.

1. Stage the three source revisions as `source/{CoordinateDescent,ProximalBase,ProximalSolvers}.jl`; develop all three into this project and instantiate.
2. Run `validate.jl`, then `worker.jl small 101 2 pilot pilot-results` under the recorded Slurm environment. Enable `BENCH_GUROBI=1` with the site Gurobi module to include licensed comparisons; otherwise HiGHS and the Julia candidates still run.
3. With validated prerequisites, submit `evaluate.sbatch` against `tasks.tsv`. Run `golden.R` after `validate.jl` creates the fixtures, using the recorded R environment.
4. Run `python3 summarize.py main-results summary` on Discovery. Run `density.jl density-results` for a separate8/32/128level reused-basis comparison.

The default design spans n1000–100000 and p100–10000 across representative dense and sparse strata. It does not attempt the full dense Cartesian grid: a100000by10000 Float64 design alone needs8GB, before LP copies, which is incompatible with the standing16GB task budget and concurrent workers. Bandwidths0.2 and0.05 assess numerical and approximation sensitivity; they are not tuned statistical recommendations.

## Additional comparisons

`admm_study.jl` checks a factorized two-block ADMM with the residual constraint divided by sqrt(n), and uses Woodbury when n is smaller than p+1. The factorization is included in path time and reused across lambda levels. The study refuses factor dimensions above2000. It compares that method, the linearized prototype and the exact dual LP on matched fresh Gaussian AR(1) inputs; its strata and seeds differ from the main study and are summarized separately.

`rbench.R` and `rbench.jl` compare quantreg's specialized Frisch–Newton lasso, conquer and Julia methods on identical fresh inputs with p100 and n1000/10000/100000. R generates centered unit sample-SD columns and writes binary Float64 data; Julia reads the same numbers. Input preparation, process startup and binary I/O are excluded from both timings. Conquer sorts a supplied lambda grid in ascending order internally; the returned coefficients are reordered before validation. Quantreg solves each level independently. The Julia LP timings retain their within-path bases. Candidate duals for quantreg are recovered at nearly-zero residuals using active-slope equalities, then made feasible by the common certificate function. The certificate's validity does not depend on that recovery being exact; an imprecise recovery can instead give a loose bound and fail the accuracy gate.

`profile.jl` warms each dispatch and samples repeated fresh paths for at least1.5seconds per method/input at two dimensions and two seeds. That gives fast methods enough snapshots to inspect. Individual slow LP or ADMM calls can run beyond the sampling target and have a2second path budget; those profiles describe sampled work, not completed-path speed. The earlier pilot's one-call profile of fast Newton had only3snapshots and is retained as a diagnostic, with no quantitative hotspot claim made from it.

Run `admm.sbatch` after the main allocation. Run `rbench.sbatch` after both the profiling and ADMM jobs. These dependencies keep concurrent memory at or below16GiB and the documented worst-case running-time allowance below2hours. Use a fresh run directory and run registry entry for a new campaign; the2026-10-25 source snapshots and outputs are immutable evidence for that campaign.

Native stopping rules differ. LP feasibility/optimality tolerances are1e-7; quantreg's matched timing study uses eps1e-7; APG uses a1e-6 KKT residual; conquer uses a1e-6 coefficient-step tolerance in timings; proximal Newton uses a1e-10 relative objective tolerance with1e-8 inner coordinate tolerance. ADMM uses a1e-5 relative primal–dual gap and1e-4 constraint residuals. Golden comparisons tighten the R tolerances. The common post-fit accuracy gates decide which timings qualify, but do not make the native stopping rules identical. Small speed differences at different achieved residuals need a tolerance-sensitivity study before making a performance claim.

The final ADMM sensitivity stage separates tuning from confirmation. It chooses among fixedrho0.01/0.1/1/10 and a residual-balancing candidate on seed901, then holds that choice fixed for seeds902–904 with two timing blocks. These candidates stop by a feasible-dual gap for the original regression objective; an auxiliary split residual is not an additional feasibility condition on the unconstrained regression coefficients. The original factorized baseline also required its split residuals to be small. Confirmation runs share one process and node, so their intervals describe input variability rather than independent machine replicates. The tuning measurements are not used as the reported confirmation speed.

`split_lp.jl` adds the standard positive/negative slope formulation, with n equality rows and nonnegative positive/negative coefficients and residuals. This avoids choosing an LP method solely from the historical absolute-value epigraph formulation. Its bounded study has a4second full-path cap, distinct from the main study's30second cap; completed matched paths can be compared within that study, while incomplete observations are not speed measurements.

The saved-fit approximation report also checks the Epanechnikov Fenchel identity. For `q=u-(tau-1/2)`, `s=2sin(asin(2q)/3)` and `u` in the check-loss subgradient interval, the convolved loss has conjugate `-3h(1-s^2)^2/16`. The report checks1206 value/derivative identities and uses projected feasible duals to bound the smoothed objective for the saved conquer fits. This separates optimization accuracy from the discrepancy between the smoothed and exact estimators; it adds no fitting runs.
