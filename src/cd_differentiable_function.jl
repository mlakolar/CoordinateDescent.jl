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
stopping rule independent of the units of the data (see [`CDOptions`](@ref)).
These functions are not exported; extend them as
`CoordinateDescent.initialize!` and so on.
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


# The value cdprox!(g, x, k, γ) would give x[k] if x[k] were v, without writing
# x: the built-in losses write x[k] only when it changes, so a coordinate that
# stays at zero never becomes a stored entry.
_cdshrink(g::ProxL1{T, Nothing}, v::T, ::Int, γ::T) where {T} = shrink(v, g.λ0 * γ)
_cdshrink(g::ProxL1{T, S}, v::T, k::Int, γ::T) where {T, S <: AbstractArray} = shrink(v, g.λ[k] * γ * g.λ0)
_cdshrink(::ProxZero, v, ::Int, _) = v

# r = y - X * x, accumulated over the columns of the stored entries of x
function _residuals!(r::AbstractVector{T}, y::AbstractVector{T}, X::AbstractMatrix{T}, x::SparseIterate{T}) where {T}
  copyto!(r, y)
  @inbounds for j = 1:nnz(x)
    k = x.nzval2ind[j]
    v = x.nzval[j]
    iszero(v) && continue
    @simd for i in eachindex(r)
      r[i] -= X[i, k] * v
    end
  end
  r
end


####################################
#
# loss |Y - X⋅β|^2 / (2⋅n)
#
####################################

"""
    CDLeastSquaresLoss(y, X)

The least squares loss `f(β) = ‖y - Xβ‖² / (2n)`. Caches the residuals
`r = y - Xβ`, so a coordinate update costs `O(n)`, and the squared column
norms `X[:, k]' X[:, k]`, computed on the first visit to each coordinate after
`initialize!`.
"""
struct CDLeastSquaresLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}
  a::Vector{T}          # squared column norms; NaN until computed

  CDLeastSquaresLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}, a::Vector{T}) where {T,S,U} =
    new(y,X,r,a)
end

function CDLeastSquaresLoss(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDLeastSquaresLoss{T, typeof(y), typeof(X)}(y,X,copy(y),fill(T(NaN), size(X, 2)))
end

numCoordinates(f::CDLeastSquaresLoss) = size(f.X, 2)

function initialize!(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  _residuals!(f.r, f.y, f.X, x)
  fill!(f.a, T(NaN))          # X may have changed in place
  nothing
end


gradient(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  - At_mul_B_row(f.X, f.r, j) / size(f.X, 1)

# X[:, k]' X[:, k] from the cache, computed and cached if needed
function _sqnorm!(f::CDLeastSquaresLoss{T}, k::Int) where {T<:AbstractFloat}
  a = f.a[k]
  if isnan(a)
    X = f.X
    a = zero(T)
    @inbounds @simd for i in axes(X, 1)
      a += X[i, k] * X[i, k]
    end
    f.a[k] = a
  end
  a
end

coordinateScale(f::CDLeastSquaresLoss, k::Int) = sqrt(_sqnorm!(f, k) / length(f.y))
responseScale(f::CDLeastSquaresLoss) = norm(f.y) / sqrt(length(f.y))

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descendCoordinate!(
  f::CDLeastSquaresLoss{T},
  g::Union{ProxL1{T}, ProxZero},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  X = f.X
  r = f.r
  n = length(f.y)

  a = f.a[k]
  b = zero(T)
  if isnan(a)
    a = zero(T)
    @inbounds @simd for i=1:n
      a += X[i, k] * X[i, k]
      b += r[i] * X[i, k]
    end
    f.a[k] = a
  else
    @inbounds @simd for i=1:n
      b += r[i] * X[i, k]
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
    CDSqrtLassoLoss(y, X)

The square-root lasso loss `f(β) = ‖y - Xβ‖`. Caches the residuals
`r = y - Xβ` and the squared column norms.

This loss is not smooth where `r = 0`, so its `descendCoordinate!` minimizes
`f + g` over a coordinate exactly instead of using a quadratic approximation.
It is implemented for [`ProxL1`](@ref ProximalBase.ProxL1) only.
"""
struct CDSqrtLassoLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}
  a::Vector{T}          # squared column norms; NaN until computed

  CDSqrtLassoLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}, a::Vector{T}) where {T,S,U} =
    new(y,X,r,a)
end

function CDSqrtLassoLoss(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDSqrtLassoLoss{T, typeof(y), typeof(X)}(y,X,copy(y),fill(T(NaN), size(X, 2)))
end

numCoordinates(f::CDSqrtLassoLoss) = size(f.X, 2)

function initialize!(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  _residuals!(f.r, f.y, f.X, x)
  fill!(f.a, T(NaN))          # X may have changed in place
  nothing
end


gradient(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  - At_mul_B_row(f.X, f.r, j) / norm(f.r)

# X[:, k]' X[:, k] from the cache, computed and cached if needed
function _sqnorm!(f::CDSqrtLassoLoss{T}, k::Int) where {T<:AbstractFloat}
  a = f.a[k]
  if isnan(a)
    X = f.X
    a = zero(T)
    @inbounds @simd for i in axes(X, 1)
      a += X[i, k] * X[i, k]
    end
    f.a[k] = a
  end
  a
end

coordinateScale(f::CDSqrtLassoLoss, k::Int) = sqrt(_sqnorm!(f, k) / length(f.y))
responseScale(f::CDSqrtLassoLoss) = norm(f.y) / sqrt(length(f.y))

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descendCoordinate!(
  f::CDSqrtLassoLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  X = f.X
  r = f.r
  n = length(f.y)

  # partial residuals with coordinate k removed: r = y - X*x + X[:,k]*x[k]
  oldVal = x[k]
  if !iszero(oldVal)
    @inbounds @simd for i=1:n
      r[i] += X[i, k] * oldVal
    end
  end

  # s = dot(r, X[:,k]), rsqr = dot(r, r), xsqr = dot(X[:,k], X[:, k]) cached
  s = zero(T)
  rsqr = zero(T)
  xsqr = f.a[k]
  if isnan(xsqr)
    xsqr = zero(T)
    @inbounds @simd for i=1:n
      xsqr += X[i, k] * X[i, k]
      s += r[i] * X[i, k]
      rsqr += r[i] * r[i]
    end
    f.a[k] = xsqr
  else
    @inbounds @simd for i=1:n
      s += r[i] * X[i, k]
      rsqr += r[i] * r[i]
    end
  end

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
    @inbounds @simd for i=1:n
      r[i] -= X[i, k] * newVal
    end
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
"""
struct CDQuadraticLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  A::S
  b::U
  Ax::Vector{T}
end

function CDQuadraticLoss(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
  issymmetric(A) || throw(ArgumentError("A must be symmetric"))
  length(b) == size(A, 2) || throw(ArgumentError("length of b must equal the number of columns of A"))
  CDQuadraticLoss{T, typeof(A), typeof(b)}(A, b, zeros(T, length(b)))
end

numCoordinates(f::CDQuadraticLoss) = length(f.b)
function initialize!(f::CDQuadraticLoss, x::SparseIterate)
  # Ax accumulated over the columns of the stored entries of x
  Ax = f.Ax
  A = f.A
  fill!(Ax, zero(eltype(Ax)))
  @inbounds for j = 1:nnz(x)
    k = x.nzval2ind[j]
    v = x.nzval[j]
    iszero(v) && continue
    @simd for i in eachindex(Ax)
      Ax[i] += A[i, k] * v
    end
  end
  nothing
end
gradient(f::CDQuadraticLoss{T}, ::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  f.Ax[j] + f.b[j]

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
