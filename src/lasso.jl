######################################################################
#
#   Lasso Solution
#
######################################################################

"""
    LassoSolution

Result returned by [`lasso`](@ref), [`sqrtLasso`](@ref),
[`scaledLasso`](@ref) and [`feasibleLasso!`](@ref).

# Fields
- `x::SparseIterate`: the estimated coefficients
- `residuals::Vector`: `y - X * x`
- `penalty::ProxL1`: the penalty at which `x` was computed
- `σ`: an estimate of the noise level (see the documentation of each solver),
  or `nothing`
"""
struct LassoSolution{T, S}
  x::SparseIterate{T}
  residuals::Vector{T}
  penalty::S
  σ::Union{T, Nothing}

  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::ProxL1{T}, σ::T) where {T, S} =
    new(x, residuals, penalty, σ)
  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::ProxL1{T}) where {T, S} =
    new(x, residuals, penalty, nothing)
end


######################################################################
#
#   Lasso Interface
#
######################################################################

"""
    lasso(X, y, λ, options=CDOptions())
    lasso(X, y, λ, ω, options=CDOptions())

Solve the lasso

    minimize  ‖y - Xβ‖² / (2n) + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent, with all `ωⱼ = 1` when the penalty loadings `ω` are
omitted. Returns a [`LassoSolution`](@ref) whose `σ` is the sample standard
deviation of the residuals.
"""
function lasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDLeastSquaresLoss(y, X)
  g = ProxL1(λ)
  coordinateDescent!(x, f, g, options)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r))
end

function lasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions()) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDLeastSquaresLoss(y, X)
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, options)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r))
end

######################################################################
#
#   Sqrt-Lasso Interface
#
######################################################################

"""
    sqrtLasso(X, y, λ, options=CDOptions(); standardizeX=true)
    sqrtLasso(X, y, λ, ω, options=CDOptions())

Solve the square-root lasso of Belloni, Chernozhukov and Wang (2011)

    minimize  ‖y - Xβ‖ + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent. In the first form the loadings are
`ωⱼ = sqrt(∑ᵢ Xᵢⱼ² / n)` when `standardizeX` is true and `ωⱼ = 1` otherwise.
The optimal `λ` does not depend on the noise level. Returns a
[`LassoSolution`](@ref) whose `σ` is the sample standard deviation of the
residuals.
"""
function sqrtLasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions();
  standardizeX=true) where {T<:AbstractFloat}

  p = size(X, 2)
  x = SparseIterate(T, p)
  f = CDSqrtLassoLoss(y, X)
  if standardizeX
    stdX = Array{T}(undef, p)
    _stdX!(stdX, X)
    g = ProxL1(λ, stdX)
  else
    g = ProxL1(λ)
  end
  coordinateDescent!(x, f, g, options)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r))
end

function sqrtLasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions()) where {T<:AbstractFloat}

  p = size(X, 2)
  x = SparseIterate(T, p)
  f = CDSqrtLassoLoss(y, X)
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, options)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r))
end


######################################################################
#
#   Scaled Lasso Interface
#
######################################################################

"""
    scaledLasso!(x, X, y, λ, ω, options=IterLassoOptions())
    scaledLasso(X, y, λ, ω, options=IterLassoOptions())

Scaled lasso of Sun and Zhang (2012), which estimates the coefficients and the
noise level jointly. It alternates between the lasso

    β̂ = argmin  ‖y - Xβ‖² / (2n) + λ σ̂ ∑ⱼ ωⱼ |βⱼ|

and the update `σ̂ = ‖y - Xβ̂‖ / √n` until the relative change in `σ̂` is below
`options.optTol` or `options.maxIter` iterations are done.

The initial `σ̂` is chosen by `options.initProcedure` (see
[`IterLassoOptions`](@ref)). `scaledLasso!` starts coordinate descent from and
overwrites `x`. Returns a [`LassoSolution`](@ref) whose `σ` is the final noise
estimate `‖y - Xβ̂‖ / √n`.
"""
function scaledLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::AbstractVector{T},
  options::IterLassoOptions=IterLassoOptions()
  )  where {T<:AbstractFloat}

  n = size(X, 1)
  f = CDLeastSquaresLoss(y,X)

  # initialize σ
  if options.initProcedure == :Screening
    σ = _findInitSigma!(X, y, options.sinit, f.r)
  elseif options.initProcedure == :InitStd
    σ = options.σinit
  elseif options.initProcedure == :WarmStart
    initialize!(f, x)
    σ = std(f.r)
  else
    throw(ArgumentError("Incorrect initialization Symbol"))
  end

  g = ProxL1(λ * σ, ω)
  for iter=1:options.maxIter
    coordinateDescent!(x, f, g, options.optionsCD)
    σnew = sqrt( sum(abs2, f.r) / n )

    if abs(σnew - σ) / σ < options.optTol
      break
    end
    σ = σnew
    g = ProxL1(λ * σ, ω)
  end

  LassoSolution{T, typeof(g)}(x, f.r, g, sqrt( sum(abs2, f.r) / n ))
end

"""
    scaledLasso(X, y, λ, ω, options=IterLassoOptions())

Same as [`scaledLasso!`](@ref), starting from a zero iterate.
"""
scaledLasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::AbstractVector{T},
  options::IterLassoOptions=IterLassoOptions()
  ) where {T<:AbstractFloat} = scaledLasso!(SparseIterate(T, size(X, 2)), X, y, λ, ω, options)


######################################################################
#
#   Feasible Lasso Interface
#
######################################################################

"""
    feasibleLasso!(x, X, y, λ0, options=IterLassoOptions())

Heteroscedasticity-robust lasso of Belloni, Chen, Chernozhukov and Hansen
(2012), which uses data-driven penalty loadings. It alternates between the
lasso

    β̂ = argmin  ‖y - Xβ‖² / (2n) + λ0 ∑ⱼ Γⱼ |βⱼ|

and the update of the loadings `Γⱼ = sqrt(∑ᵢ Xᵢⱼ² rᵢ² / n)` from the residuals
`r = y - Xβ̂`, until the relative change in `Γ` is below `options.optTol` or
`options.maxIter` iterations are done.

The initial residuals are chosen by `options.initProcedure` (see
[`IterLassoOptions`](@ref)); with `:InitStd` they come from a lasso with
penalty `λ0 σinit ∑ⱼ sⱼ |βⱼ|`, `sⱼ = sqrt(∑ᵢ Xᵢⱼ² / n)`. Coordinate descent
starts from and overwrites `x`. Returns a [`LassoSolution`](@ref) whose
`penalty.λ` holds the loadings and whose `σ` is the sample standard deviation of
the residuals.
"""
function feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ0::T,
  options::IterLassoOptions=IterLassoOptions()
  )  where {T<:AbstractFloat}

  p = size(X, 2)
  f = CDLeastSquaresLoss(y,X)
  Γ = Array{T}(undef, p)              # stores loadings
  Γold = Array{T}(undef, p)

  # initialize residuals
  if options.initProcedure == :Screening
    _findInitResiduals!(X, y, options.sinit, f.r)
  elseif options.initProcedure == :InitStd
    σ = options.σinit
    _stdX!(Γ, X)
    coordinateDescent!(x, f, ProxL1(λ0*σ, Γ), options.optionsCD)
  elseif options.initProcedure == :WarmStart
    initialize!(f, x)
  else
    throw(ArgumentError("Incorrect initialization Symbol"))
  end
  _getLoadings!(Γ, X, f.r)

  # g holds a reference to Γ, so updating Γ in place updates the penalty
  g = ProxL1(λ0, Γ)
  for iter=1:options.maxIter
    copyto!(Γold, Γ)

    coordinateDescent!(x, f, g, options.optionsCD)
    _getLoadings!(Γ, X, f.r)

    if maximum(abs.(Γold  - Γ)) / maximum(Γ) < options.optTol
      break
    end
  end

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r))
end

######################################################################
#
#   Lasso Path Interface
#
######################################################################

"""
    LassoPath(X, y, λpath, options=CDOptions(); max_hat_s=Inf, standardizeX=true)

Lasso solutions `βpath[k]` of

    minimize  ‖y - Xβ‖² / (2n) + λpath[k] ∑ⱼ ωⱼ |βⱼ|

along a decreasing sequence `λpath`, each one warm started from the previous
solution. The loadings are `ωⱼ = sqrt(∑ᵢ Xᵢⱼ² / n)` when `standardizeX` is true
and `ωⱼ = 1` otherwise.

The path stops early, after the first solution with more than `max_hat_s`
nonzero coefficients; the returned `λpath` then contains only the values that
were solved for. See also [`refitLassoPath`](@ref).
"""
struct LassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseIterate{T,1}}
end

"""
    refitLassoPath(path::LassoPath, X, y)

Least squares refits on the supports along a [`LassoPath`](@ref). Returns a
`Dict` that maps each distinct support `S` (a vector of column indices) to
`X[:, S] \\ y`.
"""
function refitLassoPath(
  path::LassoPath{T},
  X::StridedMatrix{T},
  Y::StridedVector{T}) where {T<:AbstractFloat}

  λpath = path.λpath
  βpath = path.βpath

  out = Dict{Vector{Int64},Vector{T}}()
  for i=1:length(λpath)
    S = findall(!iszero, βpath[i])
    if haskey(out, S)
      continue
    end
    out[S] = X[:, S] \ Y
  end
  out
end


function LassoPath(
  X::StridedMatrix{T},
  Y::StridedVector{T},
  λpath::Vector{T},
  options=CDOptions();
  max_hat_s=Inf, standardizeX::Bool=true) where {T<:AbstractFloat}

  p = size(X, 2)
  stdX = Array{T}(undef, p)
  if standardizeX
    _stdX!(stdX, X)
  else
    fill!(stdX, one(T))
  end

  x = SparseIterate(T, p)
  f = CDLeastSquaresLoss(Y, X)

  βpath = Vector{SparseIterate{T,1}}()
  for λ in λpath
    coordinateDescent!(x, f, ProxL1(λ, stdX), options)
    push!(βpath, copy(x))
    if nnz(x) > max_hat_s
      break
    end
  end

  LassoPath{T}(λpath[1:length(βpath)], βpath)
end
