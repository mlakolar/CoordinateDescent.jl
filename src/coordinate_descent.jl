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
descend_coordinate!(f, g, x, k) = error("quadraticApprox not implemented for $(typeof(f))")


####################################
#
# options
#
####################################


struct CDOptions
  maxIter::Int64
  optTol::Float64
end

CDOptions(;
  maxIter::Int64=2000,
  optTol::Float64=1e-7) = CDOptions(maxIter, optTol)


####################################
#
# helper functions
#
####################################

function _row_A_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::SparseVector{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())


  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  for i=1:length(nzval)
    @inbounds v += A[row, rowval[i]] * nzval[i]
  end
  v
end

function _row_A_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::StridedVector{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:p
    @inbounds v += A[row, i] * b[i]
  end
  v
end

function _row_A_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::SparseIterate{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for icoef = 1:nnz(b)
      v += A[row, b.nzval2full[icoef]] * b.nzval[icoef]
  end
  v
end


function _row_At_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::SparseVector{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  for i=1:length(nzval)
    @inbounds v += A[rowval[i], row] * nzval[i]
  end
  v
end

function _row_At_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::StridedVector{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:n
    @inbounds v += A[i, row] * b[i]
  end
  v
end


function _row_At_mul_b{T<:AbstractFloat}(A::StridedMatrix{T}, b::SparseIterate{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for icoef = 1:b.nnz
      v += A[b.nzval2full[icoef], row] * b.nzval[icoef]
  end
  v
end



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
    @inbounds r[i] = y[i] - _row_A_mul_b(X, x, i)
  end
  nothing
end


gradient{T<:AbstractFloat}(f::CDLeastSquaresLoss{T}, x::SparseIterate{T}, j::Int64) =
  _row_At_mul_b(f.X, f.r, j) / (-1*size(f.X, 1))

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descend_coordinate!{T<:AbstractFloat}(
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
    @inbounds r[i] = y[i] - _row_A_mul_b(X, x, i)
  end
  nothing
end


gradient{T<:AbstractFloat}(f::CDSqrtLassoLoss{T}, x::SparseIterate{T}, j::Int64) =
  -one(T) * _row_At_mul_b(f.X, f.r, j) / vecnorm(f.r)

# a = X[:, k]' X[:, k]
# b = X[:, k]' r
#
# h        = arg_min a/(2n) (h-b/a)^2 + λ_k⋅|x_k + h|
# xnew[k]  = arg_min a/(2n) (xnew_k - (x_k + b/a))^2 + λ_k⋅|xnew_k|
function descend_coordinate!{T<:AbstractFloat}(
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

  s = zero(T)
  xsqr = zero(T)
  rsqr = zero(T)
  @inbounds @simd for i=1:n
    xsqr += X[i, k] * X[i, k]
    s += r[i] * X[i, k]
    rsqr += r[i] * r[i]
  end

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
  _row_At_mul_b(f.A, x, j) + f.b[j]

function descend_coordinate!{T<:AbstractFloat}(
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

####################
#
# coordinate descent
#
####################

function fullPass!{T<:AbstractFloat}(
  x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1{T}, AProxL1{T}},)

  maxH = zero(T)
  for ipred = 1:length(x)
    h = descend_coordinate!(f, g, x, ipred)
    if abs(h) > maxH
      maxH = h
    end
  end
  dropzeros!(x)
  maxH
end

function nonZeroPass!{T<:AbstractFloat}(
  x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1{T}, AProxL1{T}})

  maxH = zero(T)
  for i = 1:x.nnz
    ipred = x.nzval2full[i]
    h = descend_coordinate!(f, g, x, ipred)
    if abs(h) > maxH
      maxH = h
    end
  end
  dropzeros!(x)
  maxH
end


#
# minimize f(x) + ∑ λi⋅|xi|
#
function coordinateDescent!(
  x::SparseIterate,
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, AProxL1},
  options=CDOptions())

  p = numCoordinates(f)
  if typeof(g) == AProxL1
    length(g.λ) == p || throw(DimensionMismatch())
  end

  if !iszero(x)
    initialize!(f, x)
  end

  prev_converged = false
  converged = true
  for iter=1:options.maxIter

    if converged
      maxH = fullPass!(x, f, g)
    else
      maxH = nonZeroPass!(x, f, g)
    end
    prev_converged = converged

    # test for convergence
    converged = maxH < options.optTol

    prev_converged && converged && break
  end
  x
end

coordinateDescent(f::CoordinateDifferentiableFunction, g::Union{ProxL1, AProxL1}, options=CDOptions()) =
  coordinateDescent!(SparseIterate(numCoordinates(f)), f, g, options)

##
