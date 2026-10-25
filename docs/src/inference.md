# Debiased lasso and decorrelated scores

`debiased_lasso` corrects the bias of a penalized initial estimator. For a
Gaussian linear model, the estimate of the slope vector is
`beta + theta * Xc' * (y - intercept - X * beta) / n`, where `Xc` is centered
when an intercept is fitted. Inference concerns the original regression slopes,
including those set to zero by the lasso. It does not include the intercept.

```julia
using CoordinateDescent, Random
rng = Xoshiro(77)
X = randn(rng, 200, 30)
y = X[:, 1] + randn(rng, 200)
fit = debiased_lasso(X, y; rng)   # scaled lasso initial fit and noise estimate
fit.converged || error("tighten the solver settings before inference")
coef(fit), stderror(fit), confint(fit)
joint = multiplier_bootstrap(fit; B = 1999, rng)
joint.adjusted_pvalues           # single-step Westfall--Young max-T
joint.intervals                  # simultaneous intervals
```

The default nodewise penalty is `sqrt(2log(max(p,2))/n)` on centered columns
with population standard deviation one. It is a theoretical choice, and is
**not** the pooled nodewise cross-validation rule in `hdi`. A supplied initial
penalty is on the objective `sum(abs2, residuals)/(2n) + lambda*norm(beta,1)`
on standardized columns, unless `standardize_x=false`.

## Projection and reuse

```julia
P = nodewise_lasso(X, 0.12; lambda_path = [0.3, 0.2, 0.12], gram = :auto, rng)
fit = debiased_lasso(X, y, 0.1; projection = P, rng)
Q = debiasing_matrix(X; method = :qp, mu = 0.15, rng)
fit_qp = debiased_lasso(X, y, 0.1; projection = Q, rng)
```

For nodewise lasso, a row is the vector `(e_j - gamma_j) / tau_j²`, where
`tau_j² = X_j' * (X_j - X_-j*gamma_j)/n`. The normalization includes the
penalty contribution; using only the residual mean square is incorrect.
Both the inverse and the score vectors are returned on the original scale.
`targets=[1,3,7]` computes only those rows. The nodewise regressions share one
Gram matrix, and each worker has its own state and random generator.
`lambda_path` warm starts the solve at its final penalty. With `warm_start=false`
in the solver options, that reuse is disabled.

`gram=:auto` shares the Gram matrix when its dense storage fits the specified
`max_gram_bytes` (default 1 GiB). This heuristic applies to nodewise regressions,
which reuse the matrix across many fits; it differs from the single-response
lasso heuristic. `gram=false` avoids its quadratic storage. Sparse input is
converted to a dense working design; scores use `n * length(targets)` entries.
The inverse itself is sparse. Neither the Gram matrix nor the design is copied
per predictor. Keep BLAS threads at one when parallelizing the nodewise fits.

The QP minimizes `m'G*m/2` subject to `norm(G*m-e_j,Inf) <= mu` on the working
Gram matrix `G`. Coordinate descent solves its dual
`u'G*u/2-u_j+mu*norm(u,1)`; at an optimum, `m=u` is primal feasible. The solver
checks feasibility and primal-dual gap. It does not widen an infeasible `mu`.
Rows remain asymmetric and do not have unit diagonal normalization.

A saved projection may be reused for another response on exactly the same
design, centering and scaling. The caller is responsible for this match.
For GLMs the projection depends on the fitted Hessian, so it usually cannot
be reused across responses.

## Joint inference and resampling

The multiplier bootstrap draws one common Gaussian multiplier per observation
and projects it through all influence vectors. `robust=true` in the original
fit uses centered empirical score influences; the default uses homoskedastic
model influences. `multipliers` can supply an explicit n-by-B matrix. Returned
`draws` are studentized, with targets in rows and bootstrap samples in columns.
P-values compare absolute statistics and include an add-one correction.
`stepdown=true` computes stepdown adjusted p-values. Intervals always use the
single-step maximum, so they retain simultaneous interpretation. If B is too
small to resolve the requested confidence level, the interval is unbounded.

`bootstrap_lasso(X,y,fit; B=999, wild=false, rng)` implements the separate
residual bootstrap with lasso refitting at the fixed initial penalty. It reuses
the nodewise scores, re-estimates the residual noise level in each sample,
and refits under the complete null for the max-T adjustment. It corresponds
to `hdi::boot.lasso.proj` with `boot.shortcut=true` after matching the initial
fit, nodewise tuning, scaling and sampled errors. The default uses resampled
centered residuals; `wild=true` uses normal wild errors. `resamples` supplies
the already sampled n-by-B error matrix. The bootstrap is Gaussian-only.
Its `pvalues` use absolute tails; `equal_tail_pvalues` exposes hdi's convention
for direct comparison. These differ in finite Monte Carlo samples. The centered
distributions, equal-tail p-values and max-T adjustment are compared on matched
golden inputs. This method does not repeat CV within resamples.

## GLMs and the decorrelated score

```julia
binary_y = Float64.(rand(rng, size(X,1)) .< 0.5)
glmfit = debiased_lasso(X, binary_y, 0.05; family = :binomial, rng)
test = decorrelated_score(X, binary_y, 1, 0.05; family = :binomial, null = 0, rng)
test.statistic, test.pvalue, test.converged
```

Logistic and Poisson debiasing uses the Hessian-weighted design, centered
at its weighted mean to project out an unpenalized intercept. Its score
correction uses the original covariates and `y - fitted_mean`; model variance
uses the Hessian, while the robust option uses empirical score products.
Supply the initial penalty explicitly. Binomial and Poisson dispersion is one.

The score test fits nuisance coefficients under the supplied null, then uses
Hessian-weighted penalized least squares to project the target on the nuisance
columns, following the penalized projection alternative of Ning and Liu.
Its information is the estimated Schur complement. A nonpositive value or a
degenerate Hessian is an error. `projection_lambda` is normalized by n, so its
meaning remains constant when Hessian weights have non-unit mean.

These are asymptotic procedures. Sparse coefficients, sufficient sample size,
nondegenerate information, and control of projection errors are substantive
requirements. Nodewise inference also needs suitably sparse precision rows;
the QP approach relaxes that requirement under its own assumptions. A solver
convergence flag establishes numerical convergence, not confidence coverage.
Strong correlation, dense signals, separation and poor tuning can invalidate
the normal approximation. The Discovery study records those failures without
changing the inferential claims.

## Sources and validation

The methods follow [Zhang and Zhang (2014)](https://arxiv.org/abs/1110.2563),
[van de Geer et al. (2014)](https://arxiv.org/abs/1303.0518),
[Javanmard and Montanari (2014)](https://jmlr.org/papers/v15/javanmard14a.html),
[Zhang and Cheng (2017)](https://arxiv.org/abs/1603.01295),
[Dezeure, Bühlmann and Zhang (2017)](https://arxiv.org/abs/1606.03940), and
[Ning and Liu (2017)](https://arxiv.org/abs/1412.8765).
Reference generation, replicated Discovery measurements, and coverage settings
are described in `benchmark/inference/README.md` in the source distribution.
`RESULTS.md` and `GLM_RESULTS.md` in that directory retain the observed
undercoverage and conservative score results; the default tuning is not an
empirically validated guarantee of nominal coverage across these designs.
