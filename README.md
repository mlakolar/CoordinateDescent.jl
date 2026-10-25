# HighDimensionalModels.jl

Previously CoordinateDescent.jl. Load the package with `using HighDimensionalModels`;
the package UUID and version are unchanged by this rename.

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/HighDimensionalModels.jl/dev/)
[![CI](https://github.com/mlakolar/HighDimensionalModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/HighDimensionalModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/HighDimensionalModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/HighDimensionalModels.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Lasso estimators computed by coordinate descent. The coordinate descent
itself (the passes over the coordinates and the active set, the penalty paths
with screening, the reweighting of the scaled and feasible lasso) is that of
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl), on the losses and
penalties of [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl). It
works for the ℓ1 and weighted ℓ1 penalty and for other penalties that separate
across coordinates (see [Writing a loss or a penalty](https://mlakolar.github.io/HighDimensionalModels.jl/dev/extending/)),
and by blocks for the group lasso, whose coordinates are groups.

Estimators:
* Lasso, with an intercept, standardization and penalty loadings
* Elastic net, MCP and SCAD, bounds on the coefficients, and the adaptive lasso
* Penalized logistic and Poisson regression, by proximal Newton
* Group lasso, and the multi-response lasso
* Fused lasso regression, trend filtering and the graph/generalized lasso,
  with paths, cross-validation and generalized-lasso degrees of freedom
* Square-root lasso
* Scaled lasso, from the data or from sufficient statistics (`X'X/n`, `X'y/n`,
  `y'y/n`, or a covariance matrix, one variable on the others)
* Feasible (heteroscedastic) lasso
* Inference after selection as in R's hdm: `rlasso`, `post_lasso` and
  `double_selection`
* Lasso paths, with λ chosen by cross-validation or an information criterion,
  and refitting
* Varying-coefficient models by local polynomial regression with an ℓ1 penalty

HighDimensionalModels is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| **HighDimensionalModels.jl** (this package) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Installation

Use the master checkout and prepare its dependency master checkouts:

```sh
git clone --branch master https://github.com/mlakolar/HighDimensionalModels.jl.git
cd HighDimensionalModels.jl
julia scripts/instantiate.jl
julia --project=.
```

HighDimensionalModels requires Julia 1.10 or later.

## Usage

```julia
using HighDimensionalModels

n, p = 200, 50
X = randn(n, p)
y = X[:, 1:5] * ones(5) + randn(n)

sol = lasso(X, y, 0.1; intercept=true)           # minimize ‖y - β₀ - Xβ‖² / (2n) + 0.1 ‖β‖₁
sol.x, sol.intercept                             # sparse coefficients, intercept
coef(sol), predict(sol, X)                       # StatsAPI: also residuals, nobs, dof, deviance
path = LassoPath(X, y; nλ=50, intercept=true)    # the lasso along 50 values of λ
coef(path, 0.1)                                  # the coefficients at λ = 0.1
cv = CVLassoPath(X, y; intercept=true)           # λ by 10-fold cross-validation
coef(cv)                                         # the coefficients at cv.λ1se
sol = lasso(X, y, 0.1; alpha=0.5)                # the elastic net; or penalty=MCP(1, 3), SCAD
sol = lasso(X, y, 0.1; lower=0)                  # the nonnegative lasso; also upper, per coefficient
sol = adaptive_lasso(X, y, 0.1)                   # loadings 1/|β̃ⱼ| from a first lasso
yb = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-X[:, 1])))
sol = glm_lasso(X, yb, 0.02)                      # logistic regression; also Poisson
cv = CVGLMLassoPath(X, yb)                       # λ by cross-validation of the deviance
sol = group_lasso(X, y, 0.1, [1:5, 6:10, 11:50])  # the group lasso, with orthonormalized groups
sol = sparse_group_lasso(X, y, 0.1, [1:5, 6:10, 11:50])  # groups, and variables within them
sol = multi_lasso(X, [y y.^2], 0.1)               # the same predictors for every response
sol = fused_lasso(X, y, 0.02, 0.1)               # sparsity and adjacent differences
sol = trend_filter(y, 0.01; order=1)              # piecewise linear signal fit
D = graph_difference(p, [(j,j+1) for j in 1:p-1])
path = GeneralizedLassoPath(X, y, D, [0.2, 0.1, 0.03])
bic(path), dof(path)
```

With StatsModels.jl loaded, the estimators also take a formula and a table,
and categorical variables become dummy columns (groups, for the group lasso):

```julia
using StatsModels
m = fit(LassoPath, @formula(y ~ x1 + log(x2) + region), df)
coef(m, 0.1), coefnames(m), predict(m, newdf)
```

The [documentation](https://mlakolar.github.io/HighDimensionalModels.jl/dev/) has
the mathematics of each estimator and its conventions, the options of
coordinate descent and when each pays off, the group lasso, varying-coefficient
models, how to write a loss or a penalty of your own, and the API reference.

## Examples

The scripts in [examples](examples) go through the package and its options;
the [documentation](https://mlakolar.github.io/HighDimensionalModels.jl/dev/) shows
them with their output. Run them from the root of the repository:

```
julia scripts/instantiate.jl examples
julia --project=examples examples/01_lasso.jl
julia --project=examples examples/run_all.jl      # all of them, about 20 seconds
```

The environment loads this package and its workspace dependencies from local
master checkouts. The bootstrap prepares transitive dependencies on Julia 1.10
and newer.

## Naming and the 0.8 transition

Types use `UpperCamelCase`; functions, fields and keywords use lowercase with
underscores between words. For example, use `sqrt_lasso`, `coordinate_descent!`,
and `CDOptions(; max_iter=10_000, opt_tol=1e-8, warm_start=true)`.
The previous spellings warn and forward for this release.
`coordinate_descent!` returns `CDResults`; its `minimizer` is the supplied
iterate. See [the migration table](NEWS.md#breaking-changes) for all renames.


### Quantile regression with dual variables

```julia
using HighDimensionalModels, HiGHS  # optional exact backend
path = QuantileLassoPath(X, y, [0.2, 0.1, 0.05]; tau=0.5)
beta = coef(path)
u = dual_scores(path)          # observations × penalty levels, raw solver scores
xi = -u / nobs(path)           # QR.jl sign and mean-loss normalization
```

Both exact and convolution-smoothed fits retain raw dual scores, separate
feasible certificates and KKT diagnostics. Check each fit's `converged` flag;
a time limit can return a partial exact path. See the
[primal–dual interface](docs/src/quantile.md) for scaling, weighted penalties,
zero-penalty refits and the distinction between exact and smoothed scores.
## Development tests

From the repository root, run:

```sh
julia scripts/test_master.jl
```

The script prepares an isolated test environment using dependency master
checkouts and test extras on Julia 1.10 and newer. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the naming policy and environment setup.
