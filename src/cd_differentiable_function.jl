"""
    CoordinateDifferentiableFunction

Supertype of smooth losses `f` that [`coordinateDescent!`](@ref) can minimize.
A subtype caches whatever it needs (for example residuals) to make coordinate
updates cheap, and implements

- `initialize!(f, x)`: set the cache for the iterate `x`
- `gradient(f, x, k)`: coordinate `k` of the gradient of `f` at `x`
- `numCoordinates(f)`: number of coordinates
- `descendCoordinate!(f, g, x, k)`: update `x[k]` and the cache

and optionally `coordinateScale(f, k)` and `responseScale(f)`, which make the
stopping rule independent of the units of the data (see [`CDOptions`](@ref)),
and `fullGradient!(grad, f, x)`, a faster way to compute the whole gradient.
These functions are not exported; extend them as
`CoordinateDescent.initialize!` and so on.

[`CDLeastSquaresLoss`](@ref) and [`CDQuadraticLoss`](@ref) also update whole
groups of coordinates of an `AtomIterate`, for the group lasso (see
[`coordinateDescent!`](@ref)).
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
`h = xnew[k] - xold[k]`.

Usually `h = argmin_h f(x + h eₖ) + g(x + h eₖ)`, but an implementation may
instead minimize a local quadratic approximation of `f`. The function must also
update the cache of `f` so that it is current for the new `x`.
"""
descendCoordinate!(f, g, x, k) = error("descendCoordinate not implemented for $(typeof(f))")

"""
    coordinateScale(f, k)

Scale of coordinate `k`: with `relativeTol` (see [`CDOptions`](@ref)),
[`coordinateDescent!`](@ref) measures a change `Δxₖ` as
`|Δxₖ| ⋅ coordinateScale(f, k)`. For a loss of the fitted values `Xβ` this is
the size of the change in the fitted values, for example `‖X[:, k]‖ / √n`.
Defaults to `1`.
"""
coordinateScale(f, k) = 1

"""
    responseScale(f)

Scale of the response that the scaled changes are compared with: the stopping
rule is `max_k |Δxₖ| ⋅ coordinateScale(f, k) < optTol ⋅ responseScale(f)`,
for example with `responseScale(f) = ‖y‖ / √n`. Defaults to `1`, which with the
default `coordinateScale` gives the absolute rule `max_k |Δxₖ| < optTol`.
"""
responseScale(f) = 1

"""
    fullGradient!(grad, f, x)

Write the gradient of `f` at `x` into `grad`, from the cache of `f`, which must
be current for `x`, and return `grad`. Used to check the optimality conditions
when screening (see [`LassoPath`](@ref)). Defaults to `gradient(f, x, k)` for
every coordinate.
"""
function fullGradient!(grad, f, x)
  for k in eachindex(grad)
    grad[k] = gradient(f, x, k)
  end
  grad
end


# The value cdprox!(g, x, k, γ) would give x[k] if x[k] were v, without writing
# x: the built-in losses write x[k] only when it changes, so a coordinate that
# stays at zero never becomes a stored entry.
_cdshrink(g::ProxL1{T, Nothing}, v::T, ::Int, γ::T) where {T} = shrink(v, g.λ0 * γ)
_cdshrink(g::ProxL1{T, S}, v::T, k::Int, γ::T) where {T, S <: AbstractArray} = shrink(v, g.λ[k] * γ * g.λ0)
_cdshrink(::ProxZero, v, ::Int, _) = v

# Columns centered at c, where c is the column mean when a loss fits an
# intercept and zero otherwise; a zero c takes the plain loops.

# (a, b) with b = (X[:, k] - c)' r and a = ‖X[:, k] - c‖², which is computed
# only when the cached a is NaN
@inline function _column_dot(X::AbstractMatrix{T}, r::AbstractVector{T}, k::Int, c::T, a::T) where {T}
  b = zero(T)
  if isnan(a)
    a = zero(T)
    if iszero(c)
      @inbounds @simd for i in eachindex(r)
        a += X[i, k] * X[i, k]
        b += r[i] * X[i, k]
      end
    else
      @inbounds @simd for i in eachindex(r)
        d = X[i, k] - c
        a += d * d
        b += r[i] * d
      end
    end
  elseif iszero(c)
    @inbounds @simd for i in eachindex(r)
      b += r[i] * X[i, k]
    end
  else
    @inbounds @simd for i in eachindex(r)
      b += r[i] * (X[i, k] - c)
    end
  end
  a, b
end

# ‖X[:, k] - c‖²
function _column_sqnorm(X::AbstractMatrix{T}, k::Int, c::T) where {T}
  a = zero(T)
  @inbounds @simd for i in axes(X, 1)
    a += (X[i, k] - c) * (X[i, k] - c)
  end
  a
end

# r -= h (X[:, k] - c)
@inline function _column_axpy!(r::AbstractVector{T}, X::AbstractMatrix{T}, k::Int, c::T, h::T) where {T}
  if iszero(c)
    @inbounds @simd for i in eachindex(r)
      r[i] -= X[i, k] * h
    end
  else
    @inbounds @simd for i in eachindex(r)
      r[i] -= (X[i, k] - c) * h
    end
  end
  r
end

# the entries of the active coordinates of x: the stored entries of a
# SparseIterate, the entries of the active groups of an AtomIterate
_activeEntries(x::SparseIterate) = (x.nzval2ind[j] for j in 1:nnz(x))
_activeEntries(x::AtomIterate) =
  Iterators.flatten(x.groups[ProximalBase.activeCoordinate(x, a)] for a in 1:ProximalBase.numActive(x))

# r = y - ȳ - (X - 1x̄') x, accumulated over the columns of the active entries
# of x; without xbar the columns are not centered
function _residuals!(r::AbstractVector{T}, y::AbstractVector{T}, X::AbstractMatrix{T},
                     x::Union{SparseIterate{T}, AtomIterate{T}},
                     xbar::Union{Nothing, AbstractVector{T}}=nothing, ybar::T=zero(T)) where {T}
  r .= y .- ybar
  for k in _activeEntries(x)
    v = x[k]
    iszero(v) && continue
    _column_axpy!(r, X, k, xbar === nothing ? zero(T) : xbar[k], v)
  end
  r
end

# the factor max(1 - c / (L ‖u‖), 0) by which the proximal operator of c‖⋅‖₂
# with step 1/L shrinks a vector u, given nu = ‖u‖. With L = 0 the loss does not
# depend on the group, which then goes to zero if it is penalized and keeps its
# values otherwise
function _groupShrinkFactor(nu::T, c::T, L::T) where {T<:AbstractFloat}
  iszero(L) && return iszero(c) ? one(T) : zero(T)
  nu > c / L ? one(T) - c / (L * nu) : zero(T)
end

# the change of an entry in the stopping rule, scaled as in _cdPass!
_scaledChange(f, j::Int, h, scaled::Bool) = scaled ? abs(h) * coordinateScale(f, j) : abs(h)

# the cache of group constants, one per group of an AtomIterate, unknown (NaN)
# until computed; a SparseIterate needs none
_resetGroupCache!(L::Vector{T}, ::SparseIterate) where {T} = L
_resetGroupCache!(L::Vector{T}, x::AtomIterate) where {T} =
  fill!(resize!(L, ProximalBase.numCoordinates(x)), T(NaN))


####################################
#
# loss |Y - X⋅β|^2 / (2⋅n)
#
####################################

"""
    CDLeastSquaresLoss(y, X; intercept=false)

The least squares loss `f(β) = ‖y - Xβ‖² / (2n)`. Caches the residuals
`r = y - Xβ`, so a coordinate update costs `O(n)`, and the squared column
norms `X[:, k]' X[:, k]`, computed on the first visit to each coordinate after
`initialize!`.

With `intercept`, the loss is minimized over an unpenalized intercept as well,
`f(β) = min_β₀ ‖y - β₀ - Xβ‖² / (2n)`, whose minimizer is `β₀ = ȳ - x̄'β` for
the column means `x̄ = f.xbar` and the mean `ȳ = f.ybar` of `y`. The columns
and `y` are centered on the fly, so `X` is not copied, and the residuals are
`y - β₀ - Xβ`.

On an `AtomIterate`, for the group lasso, a group `G` is updated at once, with
the largest eigenvalue of the centered `X[:, G]'X[:, G] / n` as the curvature;
it is computed when the group first leaves zero after `initialize!`.
"""
struct CDLeastSquaresLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}
  a::Vector{T}          # squared norms of the centered columns; NaN until computed
  xbar::Vector{T}       # column means with an intercept, otherwise zeros
  ybar::T               # mean of y with an intercept, otherwise zero
  L::Vector{T}          # curvatures of the groups of an AtomIterate; NaN until computed

  CDLeastSquaresLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}, a::Vector{T},
                              xbar::Vector{T}, ybar::T) where {T,S,U} =
    new(y,X,r,a,xbar,ybar,T[])
end

function CDLeastSquaresLoss(y::AbstractVector{T}, X::AbstractMatrix{T}; intercept::Bool=false) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  p = size(X, 2)
  xbar = intercept ? vec(mean(X; dims=1)) : zeros(T, p)
  ybar = intercept ? mean(y) : zero(T)
  CDLeastSquaresLoss{T, typeof(y), typeof(X)}(y,X,copy(y),fill(T(NaN), p),xbar,ybar)
end

numCoordinates(f::CDLeastSquaresLoss) = size(f.X, 2)

function initialize!(f::CDLeastSquaresLoss{T}, x::Union{SparseIterate{T}, AtomIterate{T}}) where {T<:AbstractFloat}
  _residuals!(f.r, f.y, f.X, x, f.xbar, f.ybar)
  fill!(f.a, T(NaN))          # X may have changed in place
  _resetGroupCache!(f.L, x)
  nothing
end


function gradient(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat}
  b = At_mul_B_row(f.X, f.r, j)
  iszero(f.xbar[j]) || (b -= f.xbar[j] * sum(f.r))
  - b / size(f.X, 1)
end

# -(X - 1x̄')' r / n with one product
function fullGradient!(grad::AbstractVector{T}, f::CDLeastSquaresLoss{T}, ::Union{SparseIterate{T}, AtomIterate{T}}) where {T<:AbstractFloat}
  mul!(grad, transpose(f.X), f.r)
  grad .= (f.xbar .* sum(f.r) .- grad) ./ size(f.X, 1)
end

# ‖X[:, k] - x̄ₖ‖² from the cache, computed and cached if needed
function _sqnorm!(f::CDLeastSquaresLoss{T}, k::Int) where {T<:AbstractFloat}
  a = f.a[k]
  if isnan(a)
    a = _column_sqnorm(f.X, k, f.xbar[k])
    f.a[k] = a
  end
  a
end

coordinateScale(f::CDLeastSquaresLoss, k::Int) = sqrt(_sqnorm!(f, k) / length(f.y))
function responseScale(f::CDLeastSquaresLoss{T}) where {T<:AbstractFloat}
  s = zero(T)
  @inbounds @simd for i in eachindex(f.y)
    s += (f.y[i] - f.ybar) * (f.y[i] - f.ybar)
  end
  sqrt(s / length(f.y))
end

# the intercept ȳ - x̄'x that goes with x
_intercept(f::CDLeastSquaresLoss, x::SparseIterate) = f.ybar - dot(f.xbar, x)

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
#
# with the columns centered at their means when the loss fits an intercept
function descendCoordinate!(
  f::CDLeastSquaresLoss{T},
  g::Union{ProxL1{T}, ProxZero},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  X = f.X
  r = f.r
  n = length(f.y)
  c = f.xbar[k]

  a, b = _column_dot(X, r, k, c, f.a[k])
  f.a[k] = a

  oldVal = x[k]
  newVal = _cdshrink(g, oldVal + b / a, k, n / a)
  h = newVal - oldVal
  iszero(h) && return h      # x and the residuals do not change
  x[k] = newVal

  # update internals -- residuls = y - X * xnew
  _column_axpy!(r, X, k, c, h)
  h
end

# the curvature of f over group k of x, λmax((X[:, G] - 1x̄_G')'(X[:, G] - 1x̄_G') / n),
# from the cache, computed and cached if needed
function _groupCurvature!(f::CDLeastSquaresLoss{T}, x::AtomIterate{T}, k::Int) where {T<:AbstractFloat}
  L = f.L[k]
  if isnan(L)
    G = x.groups[k]
    n = length(f.y)
    if length(G) == 1
      L = _sqnorm!(f, first(G)) / n
    else
      XG = f.X[:, G] .- transpose(f.xbar[G])
      L = max(eigmax(Symmetric(XG' * XG)) / n, zero(T))
    end
    f.L[k] = L
  end
  L
end

# group k of x ← the proximal operator of λ0 λₖ ‖⋅‖₂ with step 1/L at
# x[G] - ∇_G f(x) / L, where L is the curvature of f over G, so that the
# quadratic with curvature L majorizes f over the group (groupwise majorization
# descent, Yang and Zou, 2015); a group of one coordinate gets the step of
# descendCoordinate!. A group at zero stays there if ‖∇_G f(x)‖ ≤ λ0 λₖ,
# whatever L, which is then not needed. Returns the largest change of an entry,
# scaled with scaled; u is a buffer of at least |G| entries.
function _descendGroup!(
  f::CDLeastSquaresLoss{T},
  g::ProxGroupL2{T},
  x::AtomIterate{T},
  k::Int,
  scaled::Bool,
  u::Vector{T}) where {T<:AbstractFloat}

  X, r, n = f.X, f.r, length(f.y)
  G = x.groups[k]
  c = g.λ0 * g.λ[k]

  # u = -n ∇_G f(x), where ∇ⱼf(x) = -(X[:, j] - x̄ⱼ)' r / n
  s = zero(T)
  for (i, j) in enumerate(G)
    a, b = _column_dot(X, r, j, f.xbar[j], f.a[j])
    f.a[j] = a
    u[i] = b
    s += b * b
  end
  !ProximalBase.isActive(x, k) && sqrt(s) <= n * c && return zero(T)

  # u = x[G] - ∇_G f(x) / L
  L = _groupCurvature!(f, x, k)
  s = zero(T)
  for (i, j) in enumerate(G)
    v = iszero(L) ? x[j] : x[j] + u[i] / (n * L)
    u[i] = v
    s += v * v
  end

  t = _groupShrinkFactor(sqrt(s), c, L)
  maxH = zero(T)
  for (i, j) in enumerate(G)
    newVal = t * u[i]
    h = newVal - x[j]
    iszero(h) && continue
    x[j] = newVal
    _column_axpy!(r, X, j, f.xbar[j], h)
    maxH = max(maxH, _scaledChange(f, j, h, scaled))
  end
  maxH
end

################################################################################
#
# loss ∑_i w_i ⋅ |Y_i - X_i⋅β|^2 / (2⋅n)
#
################################################################################

"""
    CDWeightedLSLoss(y, X, w)

The weighted least squares loss `f(β) = ∑ᵢ wᵢ (yᵢ - Xᵢ'β)² / (2n)`. Caches the
residuals `r = y - Xβ` and the weighted squared column norms.

`y`, `X` and `w` are stored by reference, so changing them in place (as
[`locpolyl1`](@ref) does) changes the loss; call `initialize!` afterwards.
"""
struct CDWeightedLSLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  w::S
  r::Vector{T}
  a::Vector{T}          # weighted squared column norms; NaN until computed

  CDWeightedLSLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, r::Vector{T}, a::Vector{T}) where {T,S,U} =
    new(y,X,w,r,a)
end

function CDWeightedLSLoss(y::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) == length(w) || throw(DimensionMismatch())
  CDWeightedLSLoss{T, typeof(y), typeof(X)}(y,X,w,copy(y),fill(T(NaN), size(X, 2)))
end

numCoordinates(f::CDWeightedLSLoss) = size(f.X, 2)

function initialize!(f::CDWeightedLSLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  _residuals!(f.r, f.y, f.X, x)
  fill!(f.a, T(NaN))          # X and w may have changed in place
  nothing
end

function gradient(f::CDWeightedLSLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat}
  out = zero(T)

  n = length(f.r)
  @inbounds @simd for i=1:n
    out += f.w[i] * f.X[i, j] * f.r[i]
  end
  - out / n
end

# ∑ᵢ wᵢ X[i, k]² from the cache, computed and cached if needed
function _sqnorm!(f::CDWeightedLSLoss{T}, k::Int) where {T<:AbstractFloat}
  a = f.a[k]
  if isnan(a)
    X, w = f.X, f.w
    a = zero(T)
    @inbounds @simd for i in axes(X, 1)
      a += X[i, k] * X[i, k] * w[i]
    end
    f.a[k] = a
  end
  a
end

coordinateScale(f::CDWeightedLSLoss, k::Int) = sqrt(_sqnorm!(f, k) / length(f.y))
function responseScale(f::CDWeightedLSLoss{T}) where {T<:AbstractFloat}
  s = zero(T)
  @inbounds @simd for i in eachindex(f.y)
    s += f.w[i] * f.y[i] * f.y[i]
  end
  sqrt(s / length(f.y))
end

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descendCoordinate!(
  f::CDWeightedLSLoss{T},
  g::Union{ProxL1{T}, ProxZero},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  X = f.X
  r = f.r
  w = f.w
  n = length(f.y)

  a = f.a[k]
  b = zero(T)
  if isnan(a)
    a = zero(T)
    @inbounds @simd for i=1:n
      a += X[i, k] * X[i, k] * w[i]
      b += r[i] * X[i, k] * w[i]
    end
    f.a[k] = a
  else
    @inbounds @simd for i=1:n
      b += r[i] * X[i, k] * w[i]
    end
  end

  oldVal = x[k]
  newVal = _cdshrink(g, oldVal + b / a, k, n / a)
  h = newVal - oldVal
  iszero(h) && return h      # x and the residuals do not change
  x[k] = newVal

  # update internals -- residuls = y - X * xnew
  @inbounds @simd for i=1:n
    r[i] -= X[i, k] * h
  end
  h
end


####################################
#
# loss |Y - X⋅β|_2
#
####################################

"""
    CDSqrtLassoLoss(y, X; intercept=false)

The square-root lasso loss `f(β) = ‖y - Xβ‖`. Caches the residuals
`r = y - Xβ` and the squared column norms. With `intercept`,
`f(β) = min_β₀ ‖y - β₀ - Xβ‖`, minimized by `β₀ = ȳ - x̄'β`; as in
[`CDLeastSquaresLoss`](@ref), the columns and `y` are centered on the fly.

This loss is not smooth where `r = 0`, so its `descendCoordinate!` minimizes
`f + g` over a coordinate exactly instead of using a quadratic approximation.
It is implemented for [`ProxL1`](@ref ProximalBase.ProxL1) only.
"""
struct CDSqrtLassoLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}
  a::Vector{T}          # squared norms of the centered columns; NaN until computed
  xbar::Vector{T}       # column means with an intercept, otherwise zeros
  ybar::T               # mean of y with an intercept, otherwise zero

  CDSqrtLassoLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}, a::Vector{T},
                           xbar::Vector{T}, ybar::T) where {T,S,U} =
    new(y,X,r,a,xbar,ybar)
end

function CDSqrtLassoLoss(y::AbstractVector{T}, X::AbstractMatrix{T}; intercept::Bool=false) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  p = size(X, 2)
  xbar = intercept ? vec(mean(X; dims=1)) : zeros(T, p)
  ybar = intercept ? mean(y) : zero(T)
  CDSqrtLassoLoss{T, typeof(y), typeof(X)}(y,X,copy(y),fill(T(NaN), p),xbar,ybar)
end

numCoordinates(f::CDSqrtLassoLoss) = size(f.X, 2)

function initialize!(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  _residuals!(f.r, f.y, f.X, x, f.xbar, f.ybar)
  fill!(f.a, T(NaN))          # X may have changed in place
  nothing
end


function gradient(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat}
  b = At_mul_B_row(f.X, f.r, j)
  iszero(f.xbar[j]) || (b -= f.xbar[j] * sum(f.r))
  - b / norm(f.r)
end

function fullGradient!(grad::AbstractVector{T}, f::CDSqrtLassoLoss{T}, ::SparseIterate{T}) where {T<:AbstractFloat}
  mul!(grad, transpose(f.X), f.r)
  grad .= (f.xbar .* sum(f.r) .- grad) ./ norm(f.r)
end

# ‖X[:, k] - x̄ₖ‖² from the cache, computed and cached if needed
function _sqnorm!(f::CDSqrtLassoLoss{T}, k::Int) where {T<:AbstractFloat}
  a = f.a[k]
  if isnan(a)
    a = _column_sqnorm(f.X, k, f.xbar[k])
    f.a[k] = a
  end
  a
end

coordinateScale(f::CDSqrtLassoLoss, k::Int) = sqrt(_sqnorm!(f, k) / length(f.y))
function responseScale(f::CDSqrtLassoLoss{T}) where {T<:AbstractFloat}
  s = zero(T)
  @inbounds @simd for i in eachindex(f.y)
    s += (f.y[i] - f.ybar) * (f.y[i] - f.ybar)
  end
  sqrt(s / length(f.y))
end

_intercept(f::CDSqrtLassoLoss, x::SparseIterate) = f.ybar - dot(f.xbar, x)

# (a, s, ρ) with s = (X[:, k] - c)' r, ρ = r' r and a = ‖X[:, k] - c‖², which
# is computed only when the cached a is NaN
@inline function _column_dot_rsqr(X::AbstractMatrix{T}, r::AbstractVector{T}, k::Int, c::T, a::T) where {T}
  s = zero(T)
  ρ = zero(T)
  if isnan(a)
    a = zero(T)
    if iszero(c)
      @inbounds @simd for i in eachindex(r)
        a += X[i, k] * X[i, k]
        s += r[i] * X[i, k]
        ρ += r[i] * r[i]
      end
    else
      @inbounds @simd for i in eachindex(r)
        d = X[i, k] - c
        a += d * d
        s += r[i] * d
        ρ += r[i] * r[i]
      end
    end
  elseif iszero(c)
    @inbounds @simd for i in eachindex(r)
      s += r[i] * X[i, k]
      ρ += r[i] * r[i]
    end
  else
    @inbounds @simd for i in eachindex(r)
      s += r[i] * (X[i, k] - c)
      ρ += r[i] * r[i]
    end
  end
  a, s, ρ
end

# the exact minimizer over coordinate k, with the columns centered at their
# means when the loss fits an intercept
function descendCoordinate!(
  f::CDSqrtLassoLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  X = f.X
  r = f.r
  c = f.xbar[k]

  # partial residuals with coordinate k removed: r = y - X*x + X[:,k]*x[k]
  oldVal = x[k]
  if !iszero(oldVal)
    _column_axpy!(r, X, k, c, -oldVal)
  end

  # s = dot(r, X[:,k]), rsqr = dot(r, r), xsqr = dot(X[:,k], X[:, k]) cached
  xsqr, s, rsqr = _column_dot_rsqr(X, r, k, c, f.a[k])
  f.a[k] = xsqr

  λ = g.λ0
  if !isa(g, ProxL1{T, Nothing})
    λ *= g.λ[k]
  end

  # closed-form minimizer of ‖r - X[:,k] t‖ + λ|t| over t; zero when the
  # subgradient condition |X[:,k]'r| ≤ λ‖r‖ holds
  if abs(s) <= λ * sqrt(rsqr)
    newVal = zero(T)
  elseif s > λ * sqrt(rsqr)
    newVal = ( s - λ / sqrt(1 - λ^2 / xsqr) * sqrt(rsqr - s^2/xsqr) ) / xsqr
  else
    newVal = ( s + λ / sqrt(1 - λ^2 / xsqr) * sqrt(rsqr - s^2/xsqr) ) / xsqr
  end

  # update internals -- residuls = y - X * xnew
  if !iszero(newVal)
    _column_axpy!(r, X, k, c, newVal)
  end
  if newVal != oldVal
    x[k] = newVal
  end

  newVal - oldVal
end


####################################
#
# quadratic x'Ax/2 + x'b
#
####################################

"""
    CDQuadraticLoss(A, b)

The quadratic `f(x) = x'Ax / 2 + b'x` with symmetric `A`. Caches `Ax`, so a
coordinate update costs `O(p)`. With `A = X'X/n` and `b = -X'y/n` it gives the
same minimizer as [`CDLeastSquaresLoss`](@ref).

On an `AtomIterate`, for the group lasso, a group `G` is updated at once, with
the largest eigenvalue of `A[G, G]` as the curvature; it is computed when the
group first leaves zero after `initialize!`.
"""
struct CDQuadraticLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  A::S
  b::U
  Ax::Vector{T}
  L::Vector{T}          # curvatures of the groups of an AtomIterate; NaN until computed
end

function CDQuadraticLoss(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
  issymmetric(A) || throw(ArgumentError("A must be symmetric"))
  length(b) == size(A, 2) || throw(ArgumentError("length of b must equal the number of columns of A"))
  CDQuadraticLoss{T, typeof(A), typeof(b)}(A, b, zeros(T, length(b)), T[])
end

numCoordinates(f::CDQuadraticLoss) = length(f.b)
function initialize!(f::CDQuadraticLoss, x::Union{SparseIterate, AtomIterate})
  # Ax accumulated over the columns of the active entries of x
  Ax = f.Ax
  A = f.A
  fill!(Ax, zero(eltype(Ax)))
  for k in _activeEntries(x)
    v = x[k]
    iszero(v) && continue
    @inbounds @simd for i in eachindex(Ax)
      Ax[i] += A[i, k] * v
    end
  end
  _resetGroupCache!(f.L, x)
  nothing
end
gradient(f::CDQuadraticLoss{T}, ::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  f.Ax[j] + f.b[j]
fullGradient!(grad::AbstractVector{T}, f::CDQuadraticLoss{T}, ::Union{SparseIterate{T}, AtomIterate{T}}) where {T<:AbstractFloat} =
  grad .= f.Ax .+ f.b

# With A = X'X/n and b = -X'y/n, √A[k,k] = ‖X[:, k]‖ / √n, and
# maxₖ |bₖ| / √A[k,k] is a lower bound on ‖y‖ / √n.
coordinateScale(f::CDQuadraticLoss, k::Int) = sqrt(f.A[k, k])
function responseScale(f::CDQuadraticLoss{T}) where {T<:AbstractFloat}
  s = zero(T)
  for k in eachindex(f.b)
    a = f.A[k, k]
    if a > 0
      s = max(s, abs(f.b[k]) / sqrt(a))
    end
  end
  s
end

function descendCoordinate!(
  f::CDQuadraticLoss{T},
  g::Union{ProxL1{T}, ProxZero},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  a = f.A[k,k]
  b = gradient(f, x, k)

  oldVal = x[k]
  a = one(T) / a
  newVal = _cdshrink(g, oldVal - b * a, k, a)
  h = newVal - oldVal
  iszero(h) && return h      # x and Ax do not change
  x[k] = newVal

  # update Ax
  Ax = f.Ax
  A  = f.A
  p = length(f.Ax)
  @simd for i=1:p
    @inbounds Ax[i] += A[i,k] * h
  end

  h
end

# the curvature of f over group k of x, λmax(A[G, G]), from the cache, computed
# and cached if needed
function _groupCurvature!(f::CDQuadraticLoss{T}, x::AtomIterate{T}, k::Int) where {T<:AbstractFloat}
  L = f.L[k]
  if isnan(L)
    G = x.groups[k]
    L = length(G) == 1 ? f.A[first(G), first(G)] : eigmax(Symmetric(f.A[G, G]))
    L = max(L, zero(T))
    f.L[k] = L
  end
  L
end

# group k of x ← the proximal operator of λ0 λₖ ‖⋅‖₂ with step 1/L at
# x[G] - ∇_G f(x) / L, as for CDLeastSquaresLoss
function _descendGroup!(
  f::CDQuadraticLoss{T},
  g::ProxGroupL2{T},
  x::AtomIterate{T},
  k::Int,
  scaled::Bool,
  u::Vector{T}) where {T<:AbstractFloat}

  A, Ax, b = f.A, f.Ax, f.b
  G = x.groups[k]
  c = g.λ0 * g.λ[k]

  # u = ∇_G f(x) = (Ax + b)[G]; a group at zero stays there if ‖u‖ ≤ c
  s = zero(T)
  for (i, j) in enumerate(G)
    v = Ax[j] + b[j]
    u[i] = v
    s += v * v
  end
  !ProximalBase.isActive(x, k) && sqrt(s) <= c && return zero(T)

  # u = x[G] - ∇_G f(x) / L
  L = _groupCurvature!(f, x, k)
  s = zero(T)
  for (i, j) in enumerate(G)
    v = iszero(L) ? x[j] : x[j] - u[i] / L
    u[i] = v
    s += v * v
  end

  t = _groupShrinkFactor(sqrt(s), c, L)
  maxH = zero(T)
  for (i, j) in enumerate(G)
    newVal = t * u[i]
    h = newVal - x[j]
    iszero(h) && continue
    x[j] = newVal
    @inbounds @simd for l in eachindex(Ax)
      Ax[l] += A[l, j] * h
    end
    maxH = max(maxH, _scaledChange(f, j, h, scaled))
  end
  maxH
end
