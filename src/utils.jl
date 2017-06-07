####################################
#
# options
#
####################################

struct CDOptions
  maxIter::Int64
  optTol::Float64
  randomize::Bool           # coordinate are visitied in a randomized order or not
  warmStart::Bool           # when running CD, should we do path following or not
  numSteps::Int64           # when pathFollowing, how many points are there on the path
end

CDOptions(;
  maxIter::Int64=2000,
  optTol::Float64=1e-7,
  randomize::Bool=true,
  warmStart::Bool=true,
  numSteps::Int=50) = CDOptions(maxIter, optTol, randomize, warmStart, numSteps)


# this is used for ScaledLasso and FeasibleLasso
struct IterLassoOptions
  maxIter::Int64
  optTol::Float64
  σinit::Float64
  optionsCD::CDOptions
end

IterLassoOptions(;
  maxIter::Int64=20,
  optTol::Float64=1e-2,
  σinit::Float64=1.,
  optionsCD::CDOptions=CDOptions()) = IterLassoOptions(maxIter, optTol, σinit, optionsCD)



####################################
#
# helper functions
#
####################################


"""
Helper function that finds an initial estimate for σ that is needed
for Lasso and ScaledLasso procedures.

The procedure works as follows:

* s input variables that are most correlated with the response y are found
* y is regressed on those s features
* σ is estimated based on the residuals, which gives an upper bound on the true sigma
"""
findInitSigma(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat} =  std(findInitResiduals(X, y, s))

function findInitResiduals(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  S = findLargestCorrelations(X, y, s)

  res = X[:,S]*(X[:, S] \ y)
  @. res = y - res
  res
end

function findInitResiduals(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  S = findLargestCorrelations(w, X, y, s)

  Xs = view(X, :, S)
  res = Xs * ((Xs' * diagm(w) * Xs) \ (Xs' * diagm(w) * y))
  @. res = y - res
  res
end


# return a bit array containing indices of columns
function findLargestCorrelations(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  if s > p
    s = p
  end
  c = X' * y
  @. c = abs(c)

  # find value of s-th largest element in abs(c)
  h = binary_maxheap(T)
  @inbounds for i=1:p
      push!(h, c[i])
  end

  val = zero(T)
  for i=1:s
    val = pop!(h)
  end

  S = c .> val
end

function findLargestCorrelations(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  if s > p
    s = p
  end
  c = Array{T}(p)
  @inbounds for j=1:p
    val = zero(T)
    @simd for i=1:n
      val += X[i,j] * w[i] * y[i]
    end
    c[j] = abs(val)
  end

  # find value of s-th largest element in abs(c)
  h = binary_maxheap(T)
  @inbounds for i=1:p
      push!(h, c[i])
  end

  for i=1:s
    val = pop!(h)
  end

  S = c .> val
end


function _stdX!(out::Vector{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    v = zero(T)
    @simd for i=1:n
      v += X[i, j]^2.
    end
    out[j] = sqrt(v / n)
  end
  out
end

function _stdX!(out::Vector{T}, w::AbstractVector{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    v = zero(T)
    @simd for i=1:n
      v += w[i] * X[i, j]^2.
    end
    out[j] = sqrt(v / n)
  end
  out
end

function _getLoadings!(out::Vector{T}, X::AbstractMatrix{T}, e::AbstractVector{T}) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    v = zero(T)
    @simd for i=1:n
      v += (X[i, j]*e[i])^2.
    end
    out[j] = sqrt(v / n)
  end
  out
end
