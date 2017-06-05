abstract type CoordinateDifferentiableFunction end

"""
  Set internal parameters of the function f at the point x.
"""
initialize!(f, x) = error("initialize! not implemented for $(typeof(f))")

"""
  Coordinate k of the gradient of f evaluated at x.
"""
gradient(f, x, k) = error("gradient not implemented for $(typeof(f))")

"""
  This should return number of coordinates or blocks of coordinates over
  which the coordinate descent iterates.
"""
numCoordinates(f) = error("numCoordinates not implemented for $(typeof(f))")

"""
  Arguments:

  * f is CoordinateDifferentiableFunction
  * g is a prox function

  This function does two things:

  * It finds h such that f(x + e_k⋅h) + g(x_k + h) decreses f(x) + g(x).
    Often h = arg_min f(x + e_k⋅h) + g(x_k + h), but it could also
    minimize a local quadratic approximation.

  * The function also updates its internals. This is done by expecting
    that the algorithm to call this function again is coordinate descent.
    In future, we may want to implement other variants of coordinate descent.
"""
descendCoordinate!(f, g, x, k) = error("quadraticApprox not implemented for $(typeof(f))")


####################################
#
# loss |Y - X⋅β|^2 / (2⋅n)
#
####################################
struct CDLeastSquaresLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}

  CDLeastSquaresLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}) where {T,S,U} =
    new(y,X,r)
end

function CDLeastSquaresLoss{T<:AbstractFloat}(y::AbstractVector{T}, X::AbstractMatrix{T})
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDLeastSquaresLoss{T, typeof(y), typeof(X)}(y,X,copy(y))
end

numCoordinates(f::CDLeastSquaresLoss) = size(f.X, 2)

function initialize!{T<:AbstractFloat}(f::CDLeastSquaresLoss{T}, x::SparseIterate{T})
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


gradient{T<:AbstractFloat}(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}, j::Int64) =
  - At_mul_B_row(f.X, f.r, j) / size(f.X, 1)

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descendCoordinate!{T<:AbstractFloat}(
  f::CDLeastSquaresLoss{T},
  g::Union{ProxL1{T}, AProxL1{T}},
  x::SparseIterate{T},
  k::Int64)

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


####################################
#
# loss |Y - X⋅β|_2 / sqrt(n)
#
####################################
struct CDSqrtLassoLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  y::S
  X::U
  r::Vector{T}

  CDSqrtLassoLoss{T, S, U}(y::AbstractVector{T}, X::AbstractMatrix{T}, r::Vector{T}) where {T,S,U} =
    new(y,X,r)
end

function CDSqrtLassoLoss{T<:AbstractFloat}(y::AbstractVector{T}, X::AbstractMatrix{T})
  length(y) == size(X, 1) || throw(DimensionMismatch())
  CDSqrtLassoLoss{T, typeof(y), typeof(X)}(y,X,copy(y))
end

numCoordinates(f::CDSqrtLassoLoss) = size(f.X, 2)

function initialize!{T<:AbstractFloat}(f::CDSqrtLassoLoss{T}, x::SparseIterate{T})
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


gradient{T<:AbstractFloat}(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}, j::Int64) =
  - At_mul_B_row(f.X, f.r, j) / vecnorm(f.r)

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descendCoordinate!{T<:AbstractFloat}(
  f::CDSqrtLassoLoss{T},
  g::Union{ProxL1{T}, AProxL1{T}},
  x::SparseIterate{T},
  k::Int64)

  y = f.y
  X = f.X
  r = f.r
  n = length(f.y)

  # residuls = y - X * x + X[:, k] * x[k]
  @inbounds @simd for i=1:n
    r[i] += X[i, k] * x[k]
  end
  # r = y - X*x + X[:,k]*x[k]

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

  λ = zero(T)
  if isa(g, ProxL1{T})
    λ = g.λ
  else
    λ = g.λ[k]
  end

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
struct CDQuadraticLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  A::S
  b::U
end

function CDQuadraticLoss{T<:AbstractFloat}(A::AbstractMatrix{T}, b::AbstractVector{T})
  (issymmetric(A) && length(b) == size(A, 2)) || throw(ArgumentError())
  CDQuadraticLoss{T, typeof(A), typeof(b)}(A,b)
end

numCoordinates(f::CDQuadraticLoss) = length(f.b)
initialize!(f::CDQuadraticLoss, x::SparseIterate) = nothing
gradient{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseIterate{T}, j::Int64) =
  At_mul_B_row(f.A, x, j) + f.b[j]

function descendCoordinate!{T<:AbstractFloat}(
  f::CDQuadraticLoss{T},
  g::Union{ProxL1{T}, AProxL1{T}},
  x::SparseIterate{T},
  k::Int64)

  a = f.A[k,k]
  b = gradient(f, x, k)

  oldVal = x[k]
  a = one(T) / a
  x[k] -= b * a
  newVal = cdprox!(g, x, k, a)
  h = newVal - oldVal
end
