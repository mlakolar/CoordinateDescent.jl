# Issue78 solver study

This is an experimental benchmark and correctness suite for [issue78](https://github.com/mlakolar/CoordinateDescent.jl/issues/78). It does not yet add a supported quantile-regression API, pivotal penalty selection, CV, post-selection refits, or debiased inference.

See [QR_AUDIT.md](QR_AUDIT.md) for the historical solver choice and parametric-simplex considerations.

## Matched objectives and diagnostics

All methods fit an unpenalized intercept and the objective
`mean(rho_tau(y-a-X*b)) + lambda*sum(abs,b)`. The timing designs have columns normalized to RMS1, and use unit loadings for every method. The golden design is centered and normalized to sample SD1 to match conquer's internal standardization. `quantreg::rq.fit.lasso` minimizes a sum of losses, so it receives `n*lambda`; rqPen receives `scalex=FALSE`, `alg="br"`, and the original lambda. Conquer receives the parabolic (Epanechnikov) kernel and the same bandwidth as Julia. Its default Gaussian kernel is a different smoothing objective.

The exact dual is `max y'u/n`, with `sum(u)=0`, `tau-1<=u<=tau`, and `abs(X'u)/n<=lambda`. Certificates project the supplied dual onto the intercept/box constraints, then contract it to enforce the penalty constraints. Every returned solution retains its exact primal objective and a feasible-dual lower bound. Exact solves qualify at gap `<=1e-5*(1+abs(primal))`; negative gaps below `-1e-7` fail.

Convolution smoothing uses the existing `ProximalBase.ConvolutionCheck` and the production proximal Newton `GLMLassoPath`, plus an experimental accelerated proximal gradient implementation. A smoothed fit qualifies by a proximal KKT residual `<=2e-5`; its exact-objective gap is separately reported. At bandwidth h the Epanechnikov smoothing lies between the check loss and check loss plus `3h/16`, but that bound does not establish estimator agreement or inferential equivalence in finite samples. The ADMM candidate is a linearized two-block implementation with a power-iteration majorizer; it is one prototype, not a comparison against every ADMM variant.

## Measurement design

All fitting, tests, timing, profiling and numerical summaries run under Slurm on Discovery. Julia/BLAS/OpenMP and LP solvers use one thread. Independent workers run through GNU Parallel and exclusive srun steps, bound to allocated cores. Other workers and cluster jobs may share memory bandwidth; node identity is retained.

A worker compiles every method on smaller data of the same dense/sparse type, generates one input, then runs two independently shuffled blocks of full eight-level paths. Three seeds per stratum give independent input/process replicates; the two within-process measurements are not counted as independent inputs. Summaries use arithmetic means of per-input means, input SD, and a Student-t interval with input as the unit. The small input count and shared-node setting limit precision. Minimum times are not the decision statistic.

Measured wall time includes model construction, objective/bound updates, solver work and solution/dual extraction. A fresh LP is built for every lambda in a cold path; a warm path starts with a new model and retains it within that path. No state leaks between measured paths. LP setup, optimize and extraction times are also retained. Input generation, compilation, explicit premeasurement GC, and the external certificate pass are outside the timed region. Julia allocation/GC counts exclude most native LP solver allocations and cannot be read as total solver memory.

The LP time budget covers an entire path, not each lambda. Partial paths, solver timeouts, errors and failed certificates remain visible and are excluded from speed claims. The production Newton routine has iteration limits but no strict wall-time callback; the outer worker has a900second cap. Profiling is separate from timed observations and includes native frames where Julia's sampling profiler exposes them. Native simplex pivots and phase timings supplement those traces.

## Recreate

The run registry is `../.slurm/runs/20261025-cd78-quantile.json` relative to the workspace containing the checkout. It records exact source revisions, job IDs, resources and the staged harness versions. Remote run directory: `/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile`. The resolved Julia Manifest and R session information are retained with results.

1. Stage the three source revisions as `source/{CoordinateDescent,ProximalBase,ProximalSolvers}.jl`; develop all three into this project and instantiate.
2. Run `validate.jl`, then `worker.jl small 101 2 pilot pilot-results` under the recorded Slurm environment. Enable `BENCH_GUROBI=1` with the site Gurobi module to include licensed comparisons; otherwise HiGHS and the Julia candidates still run.
3. With validated prerequisites, submit `evaluate.sbatch` against `tasks.tsv`. Run `golden.R` after `validate.jl` creates the fixtures, using the recorded R environment.
4. Run `python3 summarize.py main-results summary` on Discovery. Run `density.jl density-results` for a separate8/32/128level reused-basis comparison.

The default design spans n1000–100000 and p100–10000 across representative dense and sparse strata. It does not attempt the full dense Cartesian grid: a100000by10000 Float64 design alone needs8GB, before LP copies, which is incompatible with the standing16GB task budget and concurrent workers. Bandwidths0.2 and0.05 assess numerical and approximation sensitivity; they are not tuned statistical recommendations.
