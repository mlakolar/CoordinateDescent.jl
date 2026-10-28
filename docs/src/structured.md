# Fused regression, trend filtering and graphs

These estimators minimize a least-squares loss divided by the number of
observations. For a general penalty matrix, the objective is

```math
\frac{1}{2n}\|y-\beta_0-X\beta\|_2^2+\lambda\|D\beta\|_1.
```

The columns keep their original units and order. Standardizing columns would
change which differences the penalty shrinks. An optional intercept is
unpenalized and estimated by centering within each training set.

```julia
sol = fused_lasso(X, y, 0.02, 0.1; intercept=true)
coef(sol), predict(sol, X), dof(sol), sol.converged
D = graph_difference(size(X, 2), [(1,2), (2,3), (1,3)])
sol = generalized_lasso(X, y, D, 0.1)
```

`fused_lasso` adds both `λ1*sum(abs, beta)` and
`λ2*sum(abs, diff(beta))`. Regression uses accelerated proximal gradient with
backtracking, gradient restart and the exact fused-lasso proximal operator of ProximalBase.
A signal fit, `fused_lasso(y, λ1, λ2)`, is one proximal step at step size `n`.

`generalized_lasso` uses the ADMM split `z = D*beta`. It factors
`X'X + rho*D'D`; the stacked matrix `[X; D]` must have full column rank for
positive penalty. A rank-deficient `X` is allowed when the penalty identifies
its null space. No ridge penalty is added. Zero penalty returns the
minimum-norm least-squares coefficients. Dense designs form dense normal
equations, so this solver is intended for moderate coefficient dimensions;
sparse signal designs and banded differences retain sparse factors.
The default `rho` accounts for the average column squared norm of the
centered design; this keeps the ADMM scale appropriate for regression as
well as identity designs. Supply `rho` explicitly for other scalings of `D`.

## Trend filtering

```julia
sol = trend_filter(y, 0.01; order=1)
sol = trend_filter(y, 0.01; order=2, x=locations)
sol = trend_filter(X, y, 0.01; order=1)  # differences of ordered coefficients
```

Order `k` penalizes differences of order `k+1`, giving piecewise polynomials
of degree `k`. Thus order zero gives piecewise constants and order one gives
piecewise linear fits. Locations must be strictly increasing; the default is
`1:p`. `difference_matrix(p, k+1; x=locations)` exposes the penalty matrix.

For positive order, the specialized ADMM of
[Ramdas and Tibshirani](https://arxiv.org/abs/1406.2082) splits the order-`k`
divided differences, then applies exact TV denoising at each iteration.
The signal solver stores sparse banded matrices and vectors, with linear
storage in signal length at fixed order. Signal fits do not allocate a dense
identity. Orders one through three use a compact banded Cholesky factor and
in-place triangular solves; other systems use the general factorization.
The default augmented parameter follows the input-spacing rule;
`rho` overrides it on the unnormalized-loss scale. Convergence checks both
primal and dual residuals, with the linear-signal rule described below.
Always inspect `converged` when changing tolerances or iteration limits.

## Paths and tuning

```julia
levels = [0.3, 0.1, 0.03, 0.01]
path = GeneralizedLassoPath(X, y, D, levels)
path = fused_lasso_path(X, y, levels; l1_ratio=0.25)
path = trend_filter_path(y, levels; order=1)
coef(path), dof(path), aic(path), bic(path)

cv = CVGeneralizedLassoPath(X, y, D, levels; nfolds=5)
cv = cv_fused_lasso(X, y, levels; l1_ratio=0.25, nfolds=5)
cv = cv_trend_filter(y, levels; order=1, nfolds=5)
coef(cv; λ=:λmin), cv.λ1se
```

Levels must be strictly decreasing, finite and nonnegative. These are fits on
a supplied grid. They do not calculate all homotopy knots, and `coef(path, λ)`
requires a fitted grid level. ADMM warm starts primal and dual variables and
reuses the factorization whenever consecutive levels have the same `rho`,
including the automatically selected value. Fused paths use
`λ1 = l1_ratio*λ` and `λ2 = (1-l1_ratio)*λ`.
Fused paths reuse centered data. Tall dense Float64 designs with at most 2,048
coefficients can also cache the normal matrix, which uses at most 32 MiB.
Convergence is checked against the original design; a failed check
switches the fit back to direct matrix products.

Double-precision linear signal fits (`order=1`) use a feasible primal-dual
objective-gap check rather than relying only on residual sizes. The gap must
be at most `0.3*reltol` times the objective, and primal/dual residuals must
also be within five times their usual bounds. Checks occur every ten
iterations, plus the first and final iteration. This can stop earlier on
large signals and require more iterations on strongly penalized cases.
Near a zero objective, where a relative gap is numerically uninformative,
the original residual test is used. Other signal orders, general designs,
Float32 fits and absolute-only tolerances (`reltol=0`) retain residual
stopping. `primal_residual` and `dual_residual` always report their actual
values at termination.

Default tolerances are calibrated by solver family. For Float64 inputs,
generalized/graph lasso uses `abstol=1e-9, reltol=1e-8`; fused regression and
other trend fits use `1e-8, 1e-7`. Linear signal fits retain `1e-7, 1e-6`
with the gap rule above. Other floating types retain `1e-7, 1e-6`.
Explicit tolerances override these defaults consistently in fits, paths and
CV folds. The stricter residual defaults address accuracy failures on
strongly penalized quadratic signals and correlated regression designs.

`converged` means the applicable stopping test passed. Except for the
linear-signal objective certificate, it reports ADMM residual convergence or
a proximal-gradient residual, not a certified objective gap. Fused regression
checks its residual at the returned coefficients using the original design,
including when normal equations are cached. No stopping rule here guarantees
a fixed relative coefficient error uniformly over ill-conditioned or
nonidentifiable designs. Tighten tolerances and check solution stability when
coefficient precision matters; changing `df_tol` also requires a sufficiently
accurate solve.

Degrees of freedom use the generalized-lasso formula
`rank(X * null(D[inactive, :]))`, plus one for an intercept, following
[Tibshirani and Taylor](https://doi.org/10.1214/11-AOS878). Inactive rows
satisfy `abs(D*beta) <= df_tol`, with default `df_tol=1e-4` in the units of
the differences. For signal trend filtering this is the number of knots plus
`order+1`; for graph signals it is the number of fused connected components.
With the additional ℓ1 penalty, components pinned to zero contribute no
degrees of freedom. Numerical classification near a transition depends on
`df_tol`; a smaller tolerance needs a correspondingly accurate solve.
For graph constraints and coordinate pins, regression uses a sparse,
orthonormal component basis and calculates its rank after multiplication by
the design. A component count alone is insufficient when the design is rank
deficient. Paths check the design rank once and reuse a degree-of-freedom
result only when the complete inactive-row pattern repeats at the next level.
The cache retains one pattern and belongs to that path. General contrasts,
numerically ambiguous edge weights and rank calculations retain the dense
null-space calculation. These choices do not change `df_tol` or solver stopping.

AIC and BIC are `n*log(RSS/n) + c*df`, with `c=2` and `c=log(n)`,
respectively. They return `NaN` for nonconverged fits and for `df >= n`.
Cross-validation uses the same grid in every fold and training-only centering.
Supply `folds` for blocked or grouped data. Signal CV keeps the complete
location grid in each fold and fits a masked observation-selection design,
so held-out responses do not enter training. Training observations must still
identify the penalty's unpenalized polynomial component. CV throws if a fit
does not converge.
Signal CV streams fold errors without computing fold degrees of freedom or
retaining fold paths/prediction matrices. The complete returned full-data path
uses direct-signal dispatch. Eligible masked systems use a diagonal observation
Gram and banded penalty factors. `max_concurrent_folds=2` bounds simultaneous
signal-fold state with threading; use one for serial fold memory. The generic
matrix-CV interface is unchanged.

The replicated CV study (archive
[`pr113-masked-cv-20261026`](https://github.com/mlakolar/HighDimensionalModels.jl/releases/download/benchmark-evidence/pr113-masked-cv-20261026.tar.gz),
report `results/cluster-20261026/RESULTS.md`) measures Float64 five/ten-fold paths through 100,000 locations for TV and linear
regular/irregular/blocked signals, and through 10,000 for quadratic signals.
Quadratic 100,000-point warmups did not complete within the study cap, and
million-point CV remains unmeasured. All completed native fits stopped below
the existing 20,000-iteration default, although the study used a 100,000 cap.
Float32 quadratic signal paths can fail the default stopping test in both
the existing and masked routes; Float64 is recommended for the measured
higher-order/large-signal range. These are scoped computational observations,
not a statistical prescription for the lambda grid, fold count or smoothing.

## Comparison with local polynomial regression

Both trend filtering and KernelSmoothing's local polynomial regression
estimate a regression function. Trend filtering selects knots through a
global difference penalty. Local polynomial regression fits a polynomial in
each kernel window, with its scale set by the bandwidth. A jump or a change in
slope can lead to different fits, so compare both on the same responses and
held-out observations. [The structured regression example](examples/11_structured.md)
computes both fits on the same data and reports their errors against a known
signal; it does not establish a general accuracy advantage for either method.

## Reference conventions and benchmarks

R's `genlasso` and `glmgen::trendfilter` use an unnormalized squared loss.
Use `lambda_R = n*lambda_Julia` on identical inputs and difference matrices.
The independent golden-data generator is `test/golden/structured/golden.R`.
The replicated timings are in the [structured study (snapshot)](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark/structured);
the `structured` group of the [benchmark harness](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md) measures the
current fits.
