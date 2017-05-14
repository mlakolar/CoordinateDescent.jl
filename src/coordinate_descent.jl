
abstract type CoordinateDifferentiableFunction{T} end

"""
  Set internal parameters of the function f at the point x.
"""
initialize!(f, x) = error("update! not implemented for $(typeof(f))")

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
update!(f, x, h, k) = error("update! not implemented for $(typeof(f))")


####################################
#
# options
#
####################################


struct CDOptions
  maxIter::Int64
  maxInnerIter::Int64
  optTol::Float64
  kktTol::Float64
end

CDOptions(;
  maxIter::Int64=2000,
  maxInnerIter::Int64=1000,
  optTol::Float64=1e-7,
  kktTol::Float64=1e-7) = CDOptions(maxIter, maxInnerIter, optTol, kktTol)


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



####################################
#
# loss |Y - X⋅β|^2 / (2⋅n)
#
####################################
struct CDLeastSquaresLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction{T}
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
function initialize!{T<:AbstractFloat}(f::CDLeastSquaresLoss{T}, x::SparseVector{T})
  # compute residuals for the loss

  X = f.X
  y = f.y
  r = f.r

  n, p = size(f.X)
  nzval = SparseArrays.nonzeros(x)
  rowval = SparseArrays.nonzeroinds(x)

  @simd for i=1:n
    @inbounds r[i] = y[i] - _row_A_mul_b(X, x, i)
  end
  nothing
end
gradient{T<:AbstractFloat}(f::CDLeastSquaresLoss{T}, x::SparseVector{T}, j::Int64) =
  _row_At_mul_b(f.X, f.r, j) / (-1*size(f.X, 1))
function quadraticApprox{T<:AbstractFloat}(
  f::CDLeastSquaresLoss{T},
  x::SparseVector{T},
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

function update!{T<:AbstractFloat}(
  f::CDLeastSquaresLoss{T},
  x::SparseVector{T},
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
struct CDQuadraticLoss{T<:AbstractFloat, S, U} <: CoordinateDifferentiableFunction{T}
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
initialize!{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseVector{T}) = nothing
gradient{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseVector{T}, j::Int64) =
  _row_At_mul_b(f.A, x, j) + f.b[j]
function quadraticApprox{T<:AbstractFloat}(
  f::CDQuadraticLoss{T},
  x::SparseVector{T},
  j::Int64)

  a = f.A[j,j]
  b = gradient(f, x, j) / a

  (a, b)
end

update!{T<:AbstractFloat}(f::CDQuadraticLoss{T}, x::SparseVector{T}, h::T, j::Int64) = nothing



####################
#
# coordinate descent
#
####################


# helper function for Active Shooting implementation of Coordinate Descent
# iterates over the active set
#
# β is a sparse vector that contains information about the active set
# when adding an element to the active set, we set that element of β to eps()
function minimize_active_set!{T<:AbstractFloat}(
  β::SparseVector{T},
  f::CoordinateDifferentiableFunction{T},
  λ::StridedVector{T},
  options::CDOptions=CDOptions())

  nzval = SparseArrays.nonzeros(β)
  rowval = SparseArrays.nonzeroinds(β)

  maxInnerIter = options.maxInnerIter
  optTol = options.optTol

  for iter=1:maxInnerIter
    fDone = true
    for j = 1:length(rowval)
      ci = rowval[j]
      # Compute the Shoot and Update the variable
      a, b = quadraticApprox(f, β, ci)
      oldVal = nzval[j]
      nzval[j] = shrink(oldVal - b, λ[ci] / a)
      h = nzval[j] - oldVal
      update!(f, β, h, ci)
      if abs(h) > optTol
        fDone = false
      end
    end
    if fDone
      break
    end
  end
  dropzeros!(β)
end

# finds index to add to the active_set
function add_violating_index!{T<:AbstractFloat}(
  β::SparseVector{T},
  f::CoordinateDifferentiableFunction{T},
  λ::StridedVector{T},
  options::CDOptions=CDOptions())

  p = numCoordinates(f)
  kktTol = options.kktTol
  nzval = SparseArrays.nonzeros(β)
  rowval = SparseArrays.nonzeroinds(β)

  val = zero(T)
  ind = 0
  for j = 1:p
    S0 = abs(gradient(f, β, j))
    if S0 > λ[j] + kktTol && S0 > val
      val = S0
      ind = j
    end
  end
  if ind != 0
    β[ind] = eps()
  end
  return ind
end

coordinateDescent{T<:AbstractFloat}(
  f::CoordinateDifferentiableFunction{T},
  λ::Vector{T},
  options::CDOptions=CDOptions()) = coordinateDescent!(spzeros(numCoordinates(f)), f, λ, options)

function coordinateDescent!{T<:AbstractFloat}(
  β::SparseVector{T},
  f::CoordinateDifferentiableFunction{T},
  λ::Vector{T},
  options::CDOptions=CDOptions())

  p = numCoordinates(f)
  length(λ) == p || throw(DimensionMismatch())

  if iszero(β)
    add_violating_index!(β, f, λ) != 0 || return β
  else
    initialize!(f, β)
  end

  for iter=1:options.maxIter
    minimize_active_set!(β, f, λ, options)
    add_violating_index!(β, f, λ) != 0 || return β
  end
  β
end
