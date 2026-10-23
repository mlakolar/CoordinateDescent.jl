# CoordinateDescent.jl

[![CI](https://github.com/mlakolar/CoordinateDescent.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/CoordinateDescent.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/CoordinateDescent.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/CoordinateDescent.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Lasso estimators computed by coordinate descent. The coordinate descent
itself (the passes over the coordinates and the active set, the penalty paths
with screening, the reweighting of the scaled and feasible lasso) is that of
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl), on the losses and
penalties of [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl). It
works for the ℓ1 and weighted ℓ1 penalty and for other penalties that separate
across coordinates (see [Writing your own loss or penalty](#writing-your-own-loss-or-penalty)),
and by blocks for the group lasso, whose coordinates are groups.

Estimators:
* Lasso, with an intercept, standardization and penalty loadings
* Group lasso
* Square-root lasso
* Scaled lasso, from the data or from sufficient statistics (`X'X/n`, `X'y/n`,
  `y'y/n`, or a covariance matrix, one variable on the others)
* Feasible (heteroscedastic) lasso
* Lasso paths and refitting
* Varying-coefficient models by local polynomial regression with an ℓ1 penalty

CoordinateDescent is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| **CoordinateDescent.jl** (this package) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Usage

```julia
using CoordinateDescent, ProximalBase

n, p = 200, 50
X = randn(n, p)
y = X[:, 1:5] * ones(5) + randn(n)

sol = lasso(X, y, 0.1)          # minimize ‖y - Xβ‖² / (2n) + 0.1 ‖β‖₁
sol.x                           # sparse coefficient vector
sol.converged, sol.iterations   # stopping rule met?, passes over the coordinates

# an unpenalized intercept (sol.intercept) and penalties scaled by the
# standard deviations of the columns
sol = lasso(X, y, 0.1; intercept=true, standardizeX=true)

# solutions along 50 values of λ, from the smallest one with a zero solution
path = LassoPath(X, y; nλ=50, intercept=true)
path.λpath, path.βpath, path.β0path

# lower-level interface
x = SparseIterate(p)
f = CDLeastSquaresLoss(y, X)
g = NormL1(0.1)
info = CDInfo()
coordinateDescent!(x, f, g, CDOptions(; optTol=1e-8), info)
info.converged, info.iterations
```

For the group lasso, `g(β) = λ Σₖ wₖ ‖β[Gₖ]‖₂`, the iterate is an
`AtomIterate` whose coordinates are the groups, which must cover all
coefficients; each step updates one group, with the largest eigenvalue of
`X[:, G]'X[:, G] / n` as the curvature (Yang and Zou, 2015):

```julia
using LinearAlgebra
groups = [1:5, 6:10, 11:20, 21:50]
g = GroupNormL2(0.1, groups)         # weights wₖ = √|Gₖ| by default
x = AtomIterate(p, groups)
coordinateDescent!(x, CDLeastSquaresLoss(y, X), g)
[norm(x[G]) for G in groups]         # the first group is selected
```

Coordinate descent stops when no coordinate changes by more than `optTol` in a
pass, the change of coefficient `k` measured in units of `‖y‖ / ‖X[:, k]‖`, so
the result does not depend on the units of the data (`CDOptions(; relativeTol =
false)` measures it in the units of the coefficients). It also stops after
`maxIter` passes (default 100 000), and then warns that the solution may be
inaccurate.

Each pass visits the coordinates in a new random order by default
(`order = :random`). The passes over the active set, which make up most of the
passes at small penalties, can be accelerated in two ways:

- **Momentum.** Each such pass starts from a point extrapolated from the last
  two (Nesterov momentum).
  - It is on by default with a random order, a `SparseIterate` and a built-in
    loss.
  - On the designs in [benchmark/README.md](benchmark/README.md) it computes
    50-λ paths 2.7–6.5 times faster than the plain random order, including
    designs whose columns share a common factor. Short solves take about as
    long either way.
  - `momentum = false` gives the plain random order of versions before 0.6.
- **Anderson acceleration.** `anderson = 5`, with `order = :cyclic`,
  extrapolates from the last passes.
  - It is faster than momentum when few passes are needed, and when n ≫ p.
  - Under a common factor it is much slower, like the cyclic order.

```julia
sol = lasso(X, y, 0.01)                                     # random order, momentum
sol = lasso(X, y, 0.01, CDOptions(; momentum = false))       # plain random order
sol = lasso(X, y, 0.01, CDOptions(; order = :cyclic, anderson = 5))
```

## Examples

The scripts in [examples](examples) go through the package and its options;
each one prints its results:

| script | shows |
|---|---|
| [01_lasso.jl](examples/01_lasso.jl) | `lasso`: the penalty level, an intercept, standardization, penalty loadings, the adaptive lasso |
| [02_solver_options.jl](examples/02_solver_options.jl) | `CDOptions` and `CDInfo`: the tolerance and its units, the order of the coordinates, warm starts and the path from λmax, `maxIter`, Float32 |
| [03_lasso_path.jl](examples/03_lasso_path.jl) | `LassoPath` and `refitLassoPath`: grids, early stopping, standardization, screening, cross-validation |
| [04_sqrt_scaled_feasible.jl](examples/04_sqrt_scaled_feasible.jl) | penalties that do not need the noise level: `sqrtLasso`, `scaledLasso`, `feasibleLasso!` and `IterLassoOptions` |
| [05_group_lasso.jl](examples/05_group_lasso.jl) | the group lasso: weights, unpenalized groups, the optimality conditions, choosing λ on a validation set |
| [06_losses.jl](examples/06_losses.jl) | `coordinateDescent!` with each loss: weighted least squares, a weighted bootstrap, a quadratic loss from `X'X` or from a covariance matrix |
| [07_varying_coefficient.jl](examples/07_varying_coefficient.jl) | `locpoly` and `locpolyl1`: kernels, bandwidths, degrees, cross-validation, standard errors |
| [08_custom_loss.jl](examples/08_custom_loss.jl) | a loss of your own: ℓ1-penalized logistic regression, checked against ProximalBase's `LogisticLoss` |
| [09_custom_penalty.jl](examples/09_custom_penalty.jl) | penalties of your own: the elastic net and the minimax concave penalty, checked against ProximalBase's |

Run them from the root of the repository:

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/01_lasso.jl
julia --project=examples examples/run_all.jl      # all of them, about 20 seconds
```

The environment loads the package from this checkout, and ProximalSolvers from
GitHub until its version 0.1.1 is registered, through `[sources]` in
[examples/Project.toml](examples/Project.toml), which Julia 1.11 and later
read. On Julia 1.10, develop the checkout and add ProximalSolvers by URL in that
environment first.

## Writing your own loss or penalty

`coordinateDescent!(x, f, g)` minimizes `f(x) + g(x)` one coordinate at a
time, by ProximalSolvers's coordinate descent: it does the passes over the
coordinates, the active set and the stopping rule, and for `NormL1` and
`GroupNormL2` also the path of penalty levels from λmax (`warmStart = false`)
and the screening with the strong rule. The loss `f` makes each coordinate
update, and the penalty `g` enters it through its proximal operator on that
coordinate. Both can be your own. A loss can also be written for ProximalSolvers
directly, as a `ProximalSolvers.CDModel` (see its documentation); the interface
below adapts to it.

### A loss

A loss is a subtype of `CoordinateDifferentiableFunction` that keeps whatever
makes an update cheap, such as the residuals or the linear predictor, and
extends these functions, which are not exported (write
`function CoordinateDescent.gradient(f::MyLoss, x, k)` and so on):

| function | |
|---|---|
| `numCoordinates(f)` | the number of coordinates |
| `initialize!(f, x)` | computes the cache for the iterate `x`, which need not be zero; called before the first pass |
| `gradient(f, x, k)` | `∂f/∂xₖ` at `x`, from the cache |
| `descendCoordinate!(f, g, x, k)` | updates `x[k]` and the cache, and returns the change of `x[k]`, which the stopping rule uses |
| `coordinateScale(f, k)`, `responseScale(f)` | optional: the scales of the coordinates and of the response, which make the stopping rule independent of the units (both 1 by default) |
| `fullGradient!(grad, f, x)` | optional: the whole gradient at once, for λmax (`gradient` for every `k` by default) |
| `f(x)` | optional: the value from the cache, which Anderson acceleration needs |

A coordinate update that works with every separable penalty minimizes a
quadratic upper bound of `f` along the coordinate plus the penalty, which is a
proximal step:

```julia
using CoordinateDescent, ProximalBase

struct MyLoss <: CoordinateDifferentiableFunction
  # the data, and a cache such as the linear predictor η = Xx
end

function CoordinateDescent.descendCoordinate!(f::MyLoss, g, x, k)
  a = curvature(f, k)                     # an upper bound on ∂²f/∂xₖ² along the coordinate
  old = x[k]
  # the proximal operator of g on entry k with step 1/a
  h = prox_entry(g, old - CoordinateDescent.gradient(f, x, k) / a, k, 1 / a) - old
  if !iszero(h)
    x[k] = old + h
    update_cache!(f, k, h)                # for example η .+= h .* X[:, k]
  end
  h
end
```

The step never increases `f + g`. With such a loss,
`coordinateDescent!(x, f, NormL1(λ, ω))` also gets the path from λmax, which
uses `gradient` and `fullGradient!`.
[examples/08_custom_loss.jl](examples/08_custom_loss.jl) builds ℓ1-penalized
logistic regression this way and checks it against Newton's method and
ProximalBase's `LogisticLoss`. On a `SymmetricSparseIterate`, whose
coordinates are the entries of the lower triangle, `ProximalBase.coordinates(x)[k]`
is coordinate `k` and `ProximalBase.prox_coordinate!(x, g, k, γ)` its proximal
step.

### A penalty

A penalty `g(x) = Σₖ gₖ(xₖ)` that separates across coordinates is a subtype
of `ProximalBase.AbstractFunction` that says it is separable, gives the
proximal operator of one entry, `ProximalBase.prox_entry(g, v, k, γ)`, which
returns `argmin_z gₖ(z) + (z - v)² / (2γ)`, and is callable for its value. For
the elastic net `λ (α ‖x‖₁ + (1 - α) ‖x‖² / 2)`:

```julia
struct MyElasticNet{T} <: ProximalBase.AbstractFunction
  λ::T
  α::T
end

ProximalBase.isseparable(::Type{<:MyElasticNet}) = true
ProximalBase.prox_entry(g::MyElasticNet, v, k, γ) = shrink(v, γ * g.λ * g.α) / (1 + γ * g.λ * (1 - g.α))
(g::MyElasticNet)(x) = g.λ * (g.α * norm(x, 1) + (1 - g.α) * sum(abs2, x) / 2)

x = SparseIterate(p)
coordinateDescent!(x, CDLeastSquaresLoss(y, X), MyElasticNet(0.1, 0.5))
```

ProximalBase has this one built in, as `ElasticNet(λα, λ(1 - α))`, with
`MCP`, `SCAD`, `NormL1Nonneg` and `IndBox`. `CDLeastSquaresLoss`,
`CDWeightedLSLoss`, `CDQuadraticLoss` and every loss whose `descendCoordinate!`
calls `prox_entry` accept such a penalty; `CDSqrtLassoLoss` and
`CDSqrtQuadraticLoss`, whose updates are the exact minimizers for the ℓ1 norm,
do not. The path from λmax and the screening need to know the penalty, so for
other penalties `warmStart = false` starts from zero and solves at `g`
directly; a path is then a loop over decreasing penalties in which each fit
starts from the previous solution. The proximal operator of a nonconvex
penalty must be defined for the steps `γ = 1/a` of the loss: `n / ‖X[:, k]‖²`
for least squares.
[examples/09_custom_penalty.jl](examples/09_custom_penalty.jl) implements the
elastic net and the minimax concave penalty (MCP). A penalty on groups of
coordinates needs a block update on an `AtomIterate`, which is implemented for
`GroupNormL2` only.
