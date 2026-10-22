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
- `iterations::Int`: passes of coordinate descent over the coordinates, summed
  over all the lasso fits the solver made
- `converged::Bool`: whether coordinate descent met its stopping rule within
  `maxIter` passes and, for [`scaledLasso!`](@ref) and [`feasibleLasso!`](@ref),
  the outer iteration met its stopping rule within its `maxIter` iterations
"""
struct LassoSolution{T, S}
  x::SparseIterate{T}
  residuals::Vector{T}
  penalty::S
  σ::Union{T, Nothing}
  iterations::Int
  converged::Bool

  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::ProxL1{T}, σ::T,
                      iterations::Integer, converged::Bool) where {T, S} =
    new(x, residuals, penalty, σ, iterations, converged)
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
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged)
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
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged)
end

function lasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions())
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), options)
end

function lasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions())
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), convert(Array{T}, ω), options)
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
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged)
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
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged)
end

function sqrtLasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions(); standardizeX=true)
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), options; standardizeX=standardizeX)
end

function sqrtLasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions())
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), convert(Array{T}, ω), options)
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
`options.optTol`, or warns after `options.maxIter` iterations.

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
  info = CDInfo()
  iterations, σconverged, change = 0, false, NaN
  for iter=1:options.maxIter
    coordinateDescent!(x, f, g, options.optionsCD, info)
    iterations += info.iterations
    σnew = sqrt( sum(abs2, f.r) / n )

    change = abs(σnew - σ) / σ
    if change < options.optTol
      σconverged = true
      break
    end
    σ = σnew
    g = ProxL1(λ * σ, ω)
  end
  σconverged || @warn "scaledLasso! did not converge in maxIter = $(options.maxIter) iterations (relative change of the noise level in the last iteration $(round(change; sigdigits = 3)) > optTol = $(options.optTol)). Increase maxIter in IterLassoOptions."

  LassoSolution{T, typeof(g)}(x, f.r, g, sqrt( sum(abs2, f.r) / n ), iterations, σconverged && info.converged)
end

scaledLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ::Real,
  ω::AbstractVector{<:Real},
  options::IterLassoOptions=IterLassoOptions()
  ) where {T<:AbstractFloat} =
  scaledLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ),
               convert(AbstractVector{T}, ω), options)

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

function scaledLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  ω::AbstractVector{<:Real}, options::IterLassoOptions=IterLassoOptions())
  T, Xf, yf = _float_data(X, y)
  scaledLasso(Xf, yf, T(λ), convert(AbstractVector{T}, ω), options)
end


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
`r = y - Xβ̂`, until the relative change in `Γ` is below `options.optTol`, or
warns after `options.maxIter` iterations.

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

  info = CDInfo()
  iterations = 0

  # initialize residuals
  if options.initProcedure == :Screening
    _findInitResiduals!(X, y, options.sinit, f.r)
  elseif options.initProcedure == :InitStd
    σ = options.σinit
    _stdX!(Γ, X)
    coordinateDescent!(x, f, ProxL1(λ0*σ, Γ), options.optionsCD, info)
    iterations += info.iterations
  elseif options.initProcedure == :WarmStart
    initialize!(f, x)
  else
    throw(ArgumentError("Incorrect initialization Symbol"))
  end
  _getLoadings!(Γ, X, f.r)

  # g holds a reference to Γ, so updating Γ in place updates the penalty
  g = ProxL1(λ0, Γ)
  Γconverged, change = false, NaN
  for iter=1:options.maxIter
    copyto!(Γold, Γ)

    coordinateDescent!(x, f, g, options.optionsCD, info)
    iterations += info.iterations
    _getLoadings!(Γ, X, f.r)

    change = maximum(abs.(Γold  - Γ)) / maximum(Γ)
    if change < options.optTol
      Γconverged = true
      break
    end
  end
  Γconverged || @warn "feasibleLasso! did not converge in maxIter = $(options.maxIter) iterations (relative change of the penalty loadings in the last iteration $(round(change; sigdigits = 3)) > optTol = $(options.optTol)). Increase maxIter in IterLassoOptions."

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), iterations, Γconverged && info.converged)
end

feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ0::Real,
  options::IterLassoOptions=IterLassoOptions()
  ) where {T<:AbstractFloat} =
  feasibleLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ0), options)

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
were solved for. The solutions are stored as `SparseVector`s, so a path takes
memory proportional to the number of nonzero coefficients. `iterations[k]` and
`converged[k]` are the passes of coordinate descent at `λpath[k]` and whether it
met its stopping rule. See also [`refitLassoPath`](@ref).
"""
struct LassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseVector{T, Int}}
  iterations::Vector{Int}
  converged::Vector{Bool}
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

  βpath = Vector{SparseVector{T, Int}}()
  iterations, converged = Int[], Bool[]
  info = CDInfo()
  for λ in λpath
    coordinateDescent!(x, f, ProxL1(λ, stdX), options, info)
    push!(βpath, _sparsevec(x))
    push!(iterations, info.iterations)
    push!(converged, info.converged)
    if nnz(x) > max_hat_s
      break
    end
  end

  LassoPath{T}(λpath[1:length(βpath)], βpath, iterations, converged)
end

function LassoPath(
  X::StridedMatrix{<:Real},
  Y::StridedVector{<:Real},
  λpath::AbstractVector{<:Real},
  options=CDOptions();
  kwargs...)
  T, Xf, Yf = _float_data(X, Y)
  LassoPath(Xf, Yf, convert(Vector{T}, λpath), options; kwargs...)
end
