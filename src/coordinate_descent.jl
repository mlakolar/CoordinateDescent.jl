#################


abstract type CDIterate{T} <: AbstractVector{T} end

mutable struct SparseIterate{T} <: CDIterate{T}
    nzval::Vector{T}         # nonzero values
    nzval2full::Vector{Int}  # Mapping from indices in nzval to full vector
    full2nzval::Vector{Int}   # Mapping from indices in full vector to indices in nzval
    nnz::Int
end

SparseIterate(n::Int) = SparseIterate{Float64}(zeros(Float64, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate{T}(::T, n::Int) = SparseIterate{T}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)

function Base.A_mul_B!{T}(out::Vector, A::Matrix, coef::SparseIterate{T})
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.nzval2full[icoef]
        c = coef.nzval[icoef]
        @simd for i = 1:size(A, 1)
            out[i] += c*A[i, ipred]
        end
    end
    out
end

function Base.dot{T}(x::Vector{T}, coef::SparseIterate{T})
    v = 0.0
    @inbounds @simd for icoef = 1:nnz(coef)
        v += x[coef.nzval2full[icoef]]*coef.nzval[icoef]
    end
    v
end

Base.length(x::SparseIterate) = length(x.full2nzval)
Base.size(x::SparseIterate) = (length(x.full2nzval),)
Base.nnz(x::SparseIterate) = x.nnz
Base.getindex{T}(x::SparseIterate{T}, ipred::Int) =
    x.full2nzval[ipred] == 0 ? zero(T) : x.nzval[x.full2nzval[ipred]]

function Base.setindex!{T}(x::SparseIterate{T}, v::T, ipred::Int)
  if x.full2nzval[ipred] == 0
    if v != zero(T)
      # newlen = length(x.nzval) + 1
      x.nnz += 1
      # resize!(x.nzval, newlen)
      # resize!(x.nzval2full, newlen)
      x.nzval[x.nnz] = v
      x.nzval2full[x.nnz] = ipred
      x.full2nzval[ipred] = x.nnz
    end
  else
    icoef = x.full2nzval[ipred]
    x.nzval[icoef] = v
  end
  v
end

Base.iszero(x::SparseIterate) = x.nnz == 0

function Base.dropzeros!{T}(x::SparseIterate{T})
  i = 1
  while i <= x.nnz
    if x.nzval[i] == zero(T)
      x.full2nzval[x.nzval2full[i]] = 0
      if i != x.nnz
        x.nzval[i] = x.nzval[x.nnz]
        x.full2nzval[x.nzval2full[x.nnz]] = i
        x.nzval2full[i] = x.nzval2full[x.nnz]
      end
      x.nnz -= 1
      i -= 1
    end
    i += 1
  end
  x
end

function Base.copy!(x::SparseIterate, y::SparseIterate)
    length(x) == length(y) || throw(DimensionMismatch())
    copy!(x.nzval, y.nzval)
    copy!(x.nzval2full, y.nzval2full)
    copy!(x.full2nzval, y.full2nzval)
    x.nnz = y.nnz
    x
end

#################



abstract type CoordinateDifferentiableFunction end

"""
  Set internal parameters of the function f at the point x.
"""
initialize!(f, x) = error("initialize! not implemented for $(typeof(f))")

"""
  Coordinate k of the gradient of f evaluated at x.
"""
gradient(f, x, k) = error("gradient not implemented for $(typeof(f))")

numCoordinates(f) = error("numCoordinates not implemented for $(typeof(f))")

"""
  Finds a and b such that
  f(x + e_k ⋅ h) = cst + (a/2)⋅(h+b)^2
"""
quadraticApprox(f, x, k) = error("quadraticApprox not implemented for $(typeof(f))")

"""
  Update internal parameters of the function f.
  This function is called after a coordinate descent at (x - e_k⋅h) with respect to coordinate k.
  The update was of size h.
  The new values is x.
"""
updateSingle!(f, x, h, k) = error("updateSingle! not implemented for $(typeof(f))")


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
# utils
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


function quadraticApprox{T<:AbstractFloat}(
  f::CDLeastSquaresLoss{T},
  x::SparseIterate{T},
  j::Int64)

  y = f.y
  X = f.X
  r = f.r
  n = length(f.y)

  a = zero(T)
  b = zero(T)
  for i=1:n
    @inbounds a += X[i, j] * X[i, j]
    @inbounds b += r[i] * X[i, j]
  end
  b = -b / a
  a = a / n

  (a, b)
end

function updateSingle!{T<:AbstractFloat}(
  f::CDLeastSquaresLoss{T},
  x::SparseIterate{T},
  h::T,
  j::Int64)

  X = f.X
  r = f.r
  n = length(r)
  @inbounds @simd for i=1:n
    r[i] -= X[i, j] * h
  end
  nothing
end

####################################
#
# quadratic x'Ax/2 + x'b
#
####################################
struct CDQuadraticLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction
  A::S
  b::U

  CDQuadraticLoss{T, S, U}(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T,S,U} =
    new(A,b)
end

function CDQuadraticLoss{T<:AbstractFloat}(A::AbstractMatrix{T}, b::AbstractVector{T})
  (issymmetric(A) && length(b) == size(A, 2)) || throw(DimensionMismatch())
  CDQuadraticLoss{T, typeof(A), typeof(b)}(A,b)
end

numCoordinates(f::CDQuadraticLoss) = length(f.b)
initialize!{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseIterate{T}) = nothing
gradient{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseIterate{T}, j::Int64) =
  _row_At_mul_b(f.A, x, j) + f.b[j]
function quadraticApprox{T<:AbstractFloat}(
  f::CDQuadraticLoss{T},
  x::SparseIterate{T},
  j::Int64)

  a = f.A[j,j]
  b = gradient(f, x, j) / a

  (a, b)
end

updateSingle!{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseIterate{T}, h::T, j::Int64) = nothing


####################
#
# coordinate descent
#
####################

function fullPass!{T<:AbstractFloat}(
  x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  λ::StridedVector{T})

  maxH = zero(T)
  for ipred = 1:length(x)
    # Compute the Shoot and Update the variable
    a, b = quadraticApprox(f, x, ipred)
    oldVal = x[ipred]
    newVal = shrink(oldVal - b, λ[ipred] / a)
    h = newVal - oldVal
    x[ipred] = newVal
    updateSingle!(f, x, h, ipred)
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
  λ::StridedVector{T})

  maxH = zero(T)
  for i = 1:x.nnz
    ipred = x.nzval2full[i]
    # Compute the Shoot and Update the variable
    a, b = quadraticApprox(f, x, ipred)
    oldVal = x[ipred]
    newVal = shrink(oldVal - b, λ[ipred] / a)
    h = newVal - oldVal
    x[ipred] = newVal
    updateSingle!(f, x, h, ipred)
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
function coordinateDescent!(x::SparseIterate, f::CoordinateDifferentiableFunction, λ::Vector, options=CDOptions())
  p = numCoordinates(f)
  length(λ) == p || throw(DimensionMismatch())

  if !iszero(x)
    initialize!(f, x)
  end

  prev_converged = false
  converged = true
  for iter=1:options.maxIter

    if converged
      # @show "fullPass"
      maxH = fullPass!(x, f, λ)
    else
      # @show "nonzeroPass"
      maxH = nonZeroPass!(x, f, λ)
    end
    prev_converged = converged

    # test for convergence
    converged = maxH < options.optTol

    prev_converged && converged && break
  end
  x
end

coordinateDescent(f::CoordinateDifferentiableFunction, λ::Vector, options=CDOptions()) =
  coordinateDescent!(SparseIterate(numCoordinates(f)), f, λ, options)

##
