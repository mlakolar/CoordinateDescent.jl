####################################
#
# options
#
####################################

"""
    CDOptions(; maxIter=100_000, optTol=1e-7, randomize=true, warmStart=true, numSteps=50)

Options for [`coordinateDescent!`](@ref).

- `maxIter`: maximum number of passes over the coordinates; reaching it without
  convergence gives a warning
- `optTol`: convergence tolerance on the largest change of a coordinate in a pass
- `randomize`: visit the coordinates in a random order in each pass, drawn from
  the global random number generator; otherwise in the order `1, 2, …`. The
  fixed order can be faster on some designs but much slower when the columns
  are strongly correlated with each other
- `warmStart`: start from the supplied iterate; otherwise start from zero and
  follow a path of decreasing penalties
- `numSteps`: number of penalty levels on that path when `warmStart` is false
"""
struct CDOptions
  maxIter::Int64
  optTol::Float64
  randomize::Bool           # coordinate are visitied in a randomized order or not
  warmStart::Bool           # when running CD, should we do path following or not
  numSteps::Int64           # when pathFollowing, how many points are there on the path
end

CDOptions(;
  maxIter::Int64=100_000,
  optTol::Float64=1e-7,
  randomize::Bool=true,
  warmStart::Bool=true,
  numSteps::Int=50) = CDOptions(maxIter, optTol, randomize, warmStart, numSteps)


"""
    CDInfo()

Convergence information that [`coordinateDescent!`](@ref) fills in when it is
passed as the last argument.

- `iterations`: number of passes over the coordinates, including those on the
  path of penalties when `warmStart` is false
- `converged`: whether the stopping rule was met before `maxIter` passes
- `maxChange`: largest change of a coordinate in the last pass
"""
mutable struct CDInfo
  iterations::Int64
  converged::Bool
  maxChange::Float64
end

CDInfo() = CDInfo(0, false, NaN)


"""
    IterLassoOptions(; maxIter=20, optTol=1e-2, initProcedure=:Screening,
                       sinit=5, σinit=1.0, optionsCD=CDOptions())

Options for the iterative procedures [`scaledLasso!`](@ref) and
[`feasibleLasso!`](@ref), which alternate a lasso fit with an update of the
noise level or of the penalty loadings.

- `maxIter`: maximum number of outer iterations
- `optTol`: tolerance on the relative change of the noise level or loadings
- `initProcedure`: how the procedure is initialized
  - `:Screening`: least squares on the `sinit` columns of `X` most correlated
    with `y`
  - `:InitStd`: noise level `σinit`
  - `:WarmStart`: the iterate passed in
- `sinit`, `σinit`: used by `:Screening` and `:InitStd`
- `optionsCD`: [`CDOptions`](@ref) for the inner lasso fits
"""
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


# Initial estimate of σ for the scaled lasso:
#
# * the s columns of X most correlated with y are found
# * y is regressed on those s columns
# * σ is the standard deviation of the residuals, which tends to overestimate
#   the true σ
_findInitSigma!(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T}) where {T <: AbstractFloat} =  std(_findInitResiduals!(X, y, s, storage))

# residuals of the least squares fit of y on the s columns of X most correlated
# with y; storage is overwritten and returned
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

# weighted version: correlations and least squares are weighted by w
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


# BitVector marking the s columns of X with the largest |X[:,j]'y|
# (more than s when there are ties)
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


# column scales out[j] = sqrt(∑ᵢ X[i,j]² / n), without centering
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

# weighted column scales out[j] = sqrt(∑ᵢ w[i] X[i,j]² / n)
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

# heteroscedasticity-robust penalty loadings out[j] = sqrt(∑ᵢ (X[i,j] e[i])² / n)
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


# weighted noise level sqrt(∑ᵢ w[i] r[i]² / ∑ᵢ w[i])
function _getSigma(w::AbstractVector{T}, r::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(w)
    σ = zero(T)
    for ii = 1:n
        σ += r[ii]^2 * w[ii]
    end
    σ /= sum(w)
    sqrt(σ)
end
