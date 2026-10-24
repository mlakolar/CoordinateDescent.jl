# Writing a loss or a penalty

[`coordinateDescent!`](@ref)`(x, f, g)` minimizes `f(x) + g(x)` one coordinate at a
time, by ProximalSolvers's coordinate descent: it does the passes over the
coordinates, the active set and the stopping rule, and for `NormL1` and
`GroupNormL2` also the path of penalty levels from λmax (`warmStart = false`)
and the screening with the strong rule. The loss `f` makes each coordinate
update, and the penalty `g` enters it through its proximal operator on that
coordinate. Both can be your own. A loss can also be written for ProximalSolvers
directly, as a `ProximalSolvers.CDModel` (see its documentation); the interface
below adapts to it.

## A loss

A loss is a subtype of [`CoordinateDifferentiableFunction`](@ref) that keeps whatever
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
[A loss of your own](examples/08_custom_loss.md) builds ℓ1-penalized
logistic regression this way and checks it against Newton's method and
ProximalBase's `LogisticLoss`. On a `SymmetricSparseIterate`, whose
coordinates are the entries of the lower triangle, `ProximalBase.coordinates(x)[k]`
is coordinate `k` and `ProximalBase.prox_coordinate!(x, g, k, γ)` its proximal
step.

## A penalty

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
`MCP`, `SCAD`, `NormL1Nonneg` and `IndBox`. [`lasso`](@ref) and
[`LassoPath`](@ref) take the built-in ones with standardization, an
intercept, ``\lambda_{\max}`` and screening, as `alpha = 0.5` or
`penalty = MCP(1, 3)` (see [The elastic net, MCP and SCAD](@ref)). `CDLeastSquaresLoss`,
`CDWeightedLSLoss`, `CDQuadraticLoss` and every loss whose `descendCoordinate!`
calls `prox_entry` accept such a penalty; `CDSqrtLassoLoss` and
`CDSqrtQuadraticLoss`, whose updates are the exact minimizers for the ℓ1 norm,
do not. The path from λmax and the screening need to know the penalty, so for
other penalties `warmStart = false` starts from zero and solves at `g`
directly; a path is then a loop over decreasing penalties in which each fit
starts from the previous solution. The proximal operator of a nonconvex
penalty must be defined for the steps `γ = 1/a` of the loss: `n / ‖X[:, k]‖²`
for least squares.
[A penalty of your own](examples/09_custom_penalty.md) implements the
elastic net and the minimax concave penalty (MCP). A penalty on groups of
coordinates needs a block update on an `AtomIterate`, which is implemented for
`GroupNormL2` only.
