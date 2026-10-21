"""
    CoordinateDifferentiableFunction

Supertype of smooth losses `f` that [`coordinateDescent!`](@ref) can minimize.
A subtype caches whatever it needs (for example residuals) to make coordinate
updates cheap, and implements

- `initialize!(f, x)`: set the cache for the iterate `x`
- `gradient(f, x, k)`: coordinate `k` of the gradient of `f` at `x`
- `numCoordinates(f)`: number of coordinates
- `descendCoordinate!(f, g, x, k)`: update `x[k]` and the cache

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


####################################
#
# loss |Y - X⋅β|^2 / (2⋅n)
#
####################################

"""
    CDLeastSquaresLoss(y, X)

The least squares loss `f(β) = ‖y - Xβ‖² / (2n)`. Caches the residuals
`r = y - Xβ`, so a coordinate update costs `O(n)`.
"""
struct CDLeastSquaresLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}

  CDLeastSquaresLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}) where {T,S,U} =
    new(y,X,r)
end

function CDLeastSquaresLoss(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDLeastSquaresLoss{T, typeof(y), typeof(X)}(y,X,copy(y))
end

numCoordinates(f::CDLeastSquaresLoss) = size(f.X, 2)

function initialize!(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  # compute residuals for the loss

  X = f.X
  y = f.y
  r = f.r

  n = size(X, 1)

  @simd for i=1:n
    @inbounds r[i] = y[i] - A_mul_B_row(X, x, i)
  end
  nothing
end


gradient(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  - At_mul_B_row(f.X, f.r, j) / size(f.X, 1)

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

  y = f.y
  X = f.X
  r = f.r
  n = length(f.y)

  a = zero(T)
  b = zero(T)
  @inbounds @simd for i=1:n
    a += X[i, k] * X[i, k]
    b += r[i] * X[i, k]
  end

  oldVal = x[k]
  x[k] += b / a
  newVal = cdprox!(g, x, k, n / a)
  h = newVal - oldVal

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
residuals `r = y - Xβ`.

`y`, `X` and `w` are stored by reference, so changing them in place (as
[`locpolyl1`](@ref) does) changes the loss; call `initialize!` afterwards.
"""
struct CDWeightedLSLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  w::S
  r::Vector{T}

  CDWeightedLSLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, r::Vector{T}) where {T,S,U} =
    new(y,X,w,r)
end

function CDWeightedLSLoss(y::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) == length(w) || throw(DimensionMismatch())
  CDWeightedLSLoss{T, typeof(y), typeof(X)}(y,X,w,copy(y))
end

numCoordinates(f::CDWeightedLSLoss) = size(f.X, 2)

function initialize!(f::CDWeightedLSLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  # compute residuals for the loss

  X = f.X
  y = f.y
  r = f.r

  n = size(X, 1)

  @simd for i=1:n
    @inbounds r[i] = y[i] - A_mul_B_row(X, x, i)
  end
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

  y = f.y
  X = f.X
  r = f.r
  w = f.w
  n = length(f.y)

  a = zero(T)
  b = zero(T)
  @inbounds @simd for i=1:n
    a += X[i, k] * X[i, k] * w[i]
    b += r[i] * X[i, k] * w[i]
  end

  oldVal = x[k]
  x[k] += b / a
  newVal = cdprox!(g, x, k, n / a)
  h = newVal - oldVal

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
`r = y - Xβ`.

This loss is not smooth where `r = 0`, so its `descendCoordinate!` minimizes
`f + g` over a coordinate exactly instead of using a quadratic approximation.
It is implemented for [`ProxL1`](@ref ProximalBase.ProxL1) only.
"""
struct CDSqrtLassoLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}

  CDSqrtLassoLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}) where {T,S,U} =
    new(y,X,r)
end

function CDSqrtLassoLoss(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDSqrtLassoLoss{T, typeof(y), typeof(X)}(y,X,copy(y))
end

numCoordinates(f::CDSqrtLassoLoss) = size(f.X, 2)

function initialize!(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}) where {T<:AbstractFloat}
  # compute residuals for the loss

  X = f.X
  y = f.y
  r = f.r

  n, p = size(f.X)

  @simd for i=1:n
    @inbounds r[i] = y[i] - A_mul_B_row(X, x, i)
  end
  nothing
end


gradient(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  - At_mul_B_row(f.X, f.r, j) / norm(f.r)

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

  y = f.y
  X = f.X
  r = f.r
  n = length(f.y)

  # partial residuals with coordinate k removed: r = y - X*x + X[:,k]*x[k]
  @inbounds @simd for i=1:n
    r[i] += X[i, k] * x[k]
  end

  s = zero(T)
  xsqr = zero(T)
  rsqr = zero(T)
  @inbounds @simd for i=1:n
    xsqr += X[i, k] * X[i, k]
    s += r[i] * X[i, k]
    rsqr += r[i] * r[i]
  end
  # s = dot(r, X[:,k])
  # xsqr = dot(X[:,k], X[:, k])
  # rsqr = dot(r, r)

  λ = g.λ0
  if !isa(g, ProxL1{T, Nothing})
    λ *= g.λ[k]
  end

  # closed-form minimizer of ‖r - X[:,k] t‖ + λ|t| over t; zero when the
  # subgradient condition |X[:,k]'r| ≤ λ‖r‖ holds
  oldVal = x[k]
  if abs(s) <= λ * sqrt(rsqr)
    x[k] = zero(T)
  elseif s > λ * sqrt(rsqr)
    x[k] = ( s - λ / sqrt(1 - λ^2 / xsqr) * sqrt(rsqr - s^2/xsqr) ) / xsqr
  else
    x[k] = ( s + λ / sqrt(1 - λ^2 / xsqr) * sqrt(rsqr - s^2/xsqr) ) / xsqr
  end

  # update internals -- residuls = y - X * xnew
  @inbounds @simd for i=1:n
    r[i] -= X[i, k] * x[k]
  end

  x[k] - oldVal
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
  Ax = f.Ax
  A = f.A
  p = length(Ax)

  @simd for i=1:p
    @inbounds Ax[i] = A_mul_B_row(A, x, i)
  end
  nothing
end
gradient(f::CDQuadraticLoss{T}, ::SparseIterate{T}, j::Int64) where {T<:AbstractFloat} =
  f.Ax[j] + f.b[j]

function descendCoordinate!(
  f::CDQuadraticLoss{T},
  g::Union{ProxL1{T}, ProxZero},
  x::SparseIterate{T},
  k::Int64) where {T<:AbstractFloat}

  a = f.A[k,k]
  b = gradient(f, x, k)

  oldVal = x[k]
  a = one(T) / a
  x[k] -= b * a
  newVal = cdprox!(g, x, k, a)
  h = newVal - oldVal

  # update Ax
  Ax = f.Ax
  A  = f.A
  p = length(f.Ax)
  @simd for i=1:p
    @inbounds Ax[i] += A[i,k] * h
  end

  h
end
