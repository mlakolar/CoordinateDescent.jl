####################################
#
# losses of coordinate descent
#
####################################

# The built-in losses are models of ProximalOPT (ProximalOPT.CDModel), the loss
# together with the cache that makes a coordinate step cheap; the constructors
# keep the names and the conventions of the earlier versions of this package.

"""
    CDLeastSquaresLoss(y, X; intercept=false)

The least squares loss `f(β) = ‖y - Xβ‖² / (2n)`, as the model of ProximalOPT
for `(1/n) * LeastSquaresLoss(y, X)` (see `ProximalOPT.cdmodel`). It caches the
residuals `r = y - Xβ`, so a coordinate update costs `O(n)`, and reads `X` and
`y` in place.

With `intercept`, the loss is minimized over an unpenalized intercept as well,
`f(β) = min_β₀ ‖y - β₀ - Xβ‖² / (2n)`, whose minimizer is `β₀ = ȳ - x̄'β` for
the column means `x̄` and the mean `ȳ` of `y`
(`ProximalOPT.intercept(f, β)`). The columns and `y` are centered on the fly,
so `X` is not copied, and the residuals are `y - β₀ - Xβ`
(`ProximalOPT.residuals(f)`).

A column that the loss does not depend on, a zero column or, with
`intercept`, a constant one, gets the minimizer of the penalty nearest zero.
`y` and `X` must not contain `NaN` or `Inf`.

On an `AtomIterate`, for the group lasso, a group `G` is updated at once, with
the largest eigenvalue of the centered `X[:, G]'X[:, G] / n` as the curvature.
"""
CDLeastSquaresLoss(y::AbstractVector, X::AbstractMatrix; intercept::Bool=false) =
  cdmodel((1 / size(X, 1)) * LeastSquaresLoss(y, X), nothing; intercept)

"""
    CDWeightedLSLoss(y, X, w)

The weighted least squares loss `f(β) = ∑ᵢ wᵢ (yᵢ - Xᵢ'β)² / (2n)`, as the
model of ProximalOPT for `(1/n) * GLMLoss(ProximalBase.Squared(), y, X;
weights = w)`. It caches the residuals `r = y - Xβ` and the weighted squared
column norms.

`y`, `X` and `w` are read in place, so changing them (as [`locpolyl1`](@ref)
does) changes the loss for the next solve. They are checked for `NaN` or `Inf`
and the weights for negative entries only when the loss is built, so build it
from initialized buffers. A coefficient whose column does not enter the loss
(`∑ᵢ wᵢ Xᵢₖ² = 0`) goes to the minimizer of the penalty nearest zero.
"""
CDWeightedLSLoss(y::AbstractVector, X::AbstractMatrix, w::AbstractVector) =
  cdmodel((1 / size(X, 1)) * GLMLoss(ProximalBase.Squared(), y, X; weights = w), nothing)

"""
    CDSqrtLassoLoss(y, X; intercept=false)

The square-root lasso loss `f(β) = ‖y - Xβ‖`, as the model of ProximalOPT for
`ProximalOPT.SqrtLeastSquaresLoss(y, X)`. It caches the residuals
`r = y - Xβ` and the squared column norms. With `intercept`,
`f(β) = min_β₀ ‖y - β₀ - Xβ‖`, minimized by `β₀ = ȳ - x̄'β`; as in
[`CDLeastSquaresLoss`](@ref), the columns and `y` are centered on the fly.

This loss is not smooth where `r = 0`, so coordinate descent minimizes `f + g`
over a coordinate exactly instead of using a quadratic approximation. It takes
the penalty `NormL1` only.
"""
CDSqrtLassoLoss(y::AbstractVector, X::AbstractMatrix; intercept::Bool=false) =
  cdmodel(SqrtLeastSquaresLoss(y, X), nothing; intercept)

"""
    CDQuadraticLoss(A, b)

The quadratic `f(x) = x'Ax / 2 + b'x` with symmetric `A`, as the model of
ProximalOPT for `QuadraticFunction(A, b)`. It caches `Ax`, so a coordinate
update costs `O(p)`. With `A = X'X/n` and `b = -X'y/n` it gives the same
minimizer as [`CDLeastSquaresLoss`](@ref).

A coordinate with `A[k, k] = 0`, which `f` does not depend on when `A` is
positive semidefinite, goes to the minimizer of the penalty nearest zero. `A`
and `b` must not contain `NaN` or `Inf`; `A` is copied into a dense matrix.

On an `AtomIterate`, for the group lasso, a group `G` is updated at once, with
the largest eigenvalue of `A[G, G]` as the curvature.
"""
function CDQuadraticLoss(A::AbstractMatrix, b::AbstractVector)
  issymmetric(A) || throw(ArgumentError("A must be symmetric"))
  cdmodel(QuadraticFunction(A, b), nothing)
end


####################################
#
# user-defined losses
#
####################################

"""
    CoordinateDifferentiableFunction

Supertype of user-defined smooth losses `f` that [`coordinateDescent!`](@ref)
can minimize. A subtype caches whatever it needs (for example residuals) to
make coordinate updates cheap, and implements

- `initialize!(f, x)`: set the cache for the iterate `x`
- `gradient(f, x, k)`: coordinate `k` of the gradient of `f` at `x`
- `numCoordinates(f)`: number of coordinates
- `descendCoordinate!(f, g, x, k)`: update coordinate `k` of `x` and the cache

and optionally `coordinateScale(f, k)` and `responseScale(f)`, which make the
stopping rule independent of the units of the data (see [`CDOptions`](@ref)),
`fullGradient!(grad, f, x)`, a faster way to compute the whole gradient, and
`f(x)`, the value of `f` at `x` from the cache, which Anderson acceleration
needs (`anderson` in [`CDOptions`](@ref)). These functions are not exported;
extend them as `CoordinateDescent.initialize!` and so on.

The penalty `g` enters only through `descendCoordinate!`. A loss whose step
ends with the proximal operator of `g` on coordinate `k`,
`ProximalBase.prox_coordinate!(x, g, k, γ)`, works with every separable penalty
of ProximalBase and with user-defined ones (see [`coordinateDescent!`](@ref)).

Coordinate descent runs on ProximalOPT, which takes such a loss through a
model (`ProximalOPT.CDModel`); a loss written for ProximalOPT directly, as a
`CDModel`, is also accepted.
"""
abstract type CoordinateDifferentiableFunction end

"""
    initialize!(f, x)

Set the internal cache of `f` (for example the residuals) for the iterate `x`.
Called by [`coordinateDescent!`](@ref) before the first pass.
"""
initialize!(f, x) = error("initialize! not implemented for $(typeof(f))")

"""
    gradient(f, x, k)

Coordinate `k` of the gradient of `f` at `x`, computed from the cache of `f`,
which must be current for `x`.
"""
gradient(f, x, k) = error("gradient not implemented for $(typeof(f))")

"""
    numCoordinates(f)

Number of coordinates (or blocks of coordinates) over which coordinate descent
iterates.
"""
numCoordinates(f) = error("numCoordinates not implemented for $(typeof(f))")

"""
    descendCoordinate!(f, g, x, k)

Update coordinate `k` of `x` to decrease `f(x) + g(x)`, and return the change
of the coordinate. The cache of `f` must be updated for the new `x`.
"""
descendCoordinate!(f, g, x, k) = error("descendCoordinate! not implemented for $(typeof(f)) and $(typeof(g))")

"""
    coordinateScale(f, k)

The scale of coordinate `k` for the relative stopping rule (see `optTol` in
[`CDOptions`](@ref)): the change of `x[k]` is multiplied by it. Defaults to 1.
"""
coordinateScale(f, k) = 1

"""
    responseScale(f)

The scale that `optTol` is multiplied by in the relative stopping rule (see
[`CDOptions`](@ref)). Defaults to 1.
"""
responseScale(f) = 1

"""
    fullGradient!(grad, f, x)

Write the gradient of `f` at `x` into `grad`, from the cache of `f`, which must
be current for `x`, and return `grad`. Used to find the largest useful penalty
level when `warmStart` is false. Defaults to `gradient(f, x, k)` for every
coordinate.
"""
function fullGradient!(grad, f, x)
  for k in eachindex(grad)
    grad[k] = gradient(f, x, k)
  end
  grad
end

# The model of ProximalOPT for a user-defined loss: each coordinate step is the
# loss's descendCoordinate!, and the optional functions give the scales of the
# stopping rule, the value and the gradient
struct _LossModel{T, F<:CoordinateDifferentiableFunction} <: CDModel{T}
  f::F
end

_LossModel(f::CoordinateDifferentiableFunction, x) = _LossModel{float(eltype(x)), typeof(f)}(f)

function ProximalOPT.initialize!(m::_LossModel, x)
  numCoordinates(m.f) == ncoordinates(x) || throw(DimensionMismatch(
    "the loss has $(numCoordinates(m.f)) coordinates and the iterate $(ncoordinates(x))"))
  initialize!(m.f, x)
  m
end

ProximalOPT.coordinate_step!(m::_LossModel{T}, g, x::SparseIterate{T}, k::Int) where {T} =
  descendCoordinate!(m.f, g, x, k)
ProximalOPT.coordinate_step!(m::_LossModel{T}, g, x::SymmetricSparseIterate{T}, k::Int) where {T} =
  descendCoordinate!(m.f, g, x, k)

ProximalOPT.coordinate_scale(m::_LossModel{T}, x, k::Int) where {T} = T(coordinateScale(m.f, k))
ProximalOPT.response_scale(m::_LossModel{T}, x) where {T} = T(responseScale(m.f))

# the value from the cache when the loss is callable, NaN otherwise
ProximalOPT.loss_value(m::_LossModel{T}, x) where {T} = applicable(m.f, x) ? T(m.f(x)) : T(NaN)

ProximalOPT.gradient!(G::AbstractVector, m::_LossModel, x::SparseIterate) = fullGradient!(G, m.f, x)

# the gradient over the entries of the p × p matrix, from the coordinates of
# the lower triangle: λmax compares |G[r, c]| with the weight of (r, c), as
# |gradient(f, x, k)| was compared with the weight of coordinate k before
function ProximalOPT.gradient!(G::AbstractMatrix, m::_LossModel, x::SymmetricSparseIterate)
  for k in 1:ncoordinates(x)
    r, c = tril_cartesian(x.p, k)
    G[r, c] = G[c, r] = gradient(m.f, x, k)
  end
  G
end
