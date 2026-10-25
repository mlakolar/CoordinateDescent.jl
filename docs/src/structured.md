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
`λ2*sum(abs, diff(beta))`. Regression uses proximal gradient with
backtracking and the exact fused-lasso proximal operator of ProximalBase.
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
identity. The default augmented parameter follows the input-spacing rule;
`rho` overrides it on the unnormalized-loss scale. Both ADMM residual tests
must pass before `converged` becomes true. Always inspect this field when
changing tolerances or iteration limits.

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
requires a fitted grid level. ADMM warm starts primal and dual variables;
setting a common `rho` also reuses the factorization. Fused paths use
`λ1 = l1_ratio*λ` and `λ2 = (1-l1_ratio)*λ`.

Degrees of freedom use the generalized-lasso formula
`rank(X * null(D[inactive, :]))`, plus one for an intercept, following
[Tibshirani and Taylor](https://doi.org/10.1214/11-AOS878). Inactive rows
satisfy `abs(D*beta) <= df_tol`, with default `df_tol=1e-4` in the units of
the differences. For signal trend filtering this is the number of knots plus
`order+1`; for graph signals it is the number of fused connected components.
With the additional ℓ1 penalty, components pinned to zero contribute no
degrees of freedom. Numerical classification near a transition depends on
`df_tol`; a smaller tolerance needs a correspondingly accurate solve.
Generic designs use a dense null-space calculation for degrees of freedom.

AIC and BIC are `n*log(RSS/n) + c*df`, with `c=2` and `c=log(n)`,
respectively. They return `NaN` for nonconverged fits and for `df >= n`.
Cross-validation uses the same grid in every fold and training-only centering.
Supply `folds` for blocked or grouped data. Signal CV keeps the complete
location grid in each fold and fits a sparse observation-selection design,
so held-out responses do not enter training. Training observations must still
identify the penalty's unpenalized polynomial component. CV throws if a fit
does not converge.

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
The independent golden-data generator and replicated Discovery launcher are
in [`benchmark/structured`](https://github.com/mlakolar/CoordinateDescent.jl/tree/master/benchmark/structured).
