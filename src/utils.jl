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
  initProcedure::Symbol    # :Screening, :InitStd, :WarmStart
  sinit::Int64             # how many columns of X to use to estimate initial variance or obtain initial residuals
  σinit::Float64
  optionsCD::CDOptions
end

IterLassoOptions(;
  maxIter::Int64=20,
  optTol::Float64=1e-2,
  initProcedure::Symbol=:Screening,
  sinit::Int64=5,
  σinit::Float64=1.,
  optionsCD::CDOptions=CDOptions()) = IterLassoOptions(maxIter, optTol, initProcedure, sinit, σinit, optionsCD)



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
_findInitSigma!(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T}) where {T <: AbstractFloat} =  std(_findInitResiduals!(X, y, s, storage))

function _findInitResiduals!(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T}) where {T <: AbstractFloat}

  S = _findLargestCorrelations(X, y, s)
  Xs = view(X, :, S)
  mul!(storage, Xs, Xs \ y)
  @. storage = y - storage
  return storage
end

function _findInitResiduals!(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T}) where {T <: AbstractFloat}

  S = _findLargestCorrelations(w, X, y, s)

  Xs = view(X, :, S)
  mul!(storage, Xs, (Xs' * Diagonal(w) * Xs) \ (Xs' * Diagonal(w) * y))
  @. storage = y - storage
  return storage
end


# return a bit array containing indices of columns
function _findLargestCorrelations(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  p = size(X, 2)
  storage = Array{T}(undef, p)
  mul!(storage, transpose(X), y)
  @. storage = abs(storage)
  S = storage .>= nlargest(s, storage)[end]
end

function _findLargestCorrelations(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  storage = Array{T}(undef, p)
  @inbounds for j=1:p
    val = zero(T)
    @simd for i=1:n
      val += X[i,j] * w[i] * y[i]
    end
    storage[j] = abs(val)
  end
  S = storage .>= nlargest(s, storage)[end]
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


function _getSigma(w::AbstractVector{T}, r::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(w)
    σ = zero(T)
    for ii = 1:n
        σ += r[ii]^2 * w[ii]
    end
    σ /= sum(w)
    sqrt(σ)
end
