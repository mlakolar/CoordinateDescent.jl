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
- `residuals::Vector`: `y - intercept - X * x`
- `penalty::ProxL1`: the penalty at which `x` was computed
- `σ`: an estimate of the noise level (see the documentation of each solver),
  or `nothing`
- `iterations::Int`: passes of coordinate descent over the coordinates, summed
  over all the lasso fits the solver made
- `converged::Bool`: whether coordinate descent met its stopping rule within
  `maxIter` passes and, for [`scaledLasso!`](@ref) and [`feasibleLasso!`](@ref),
  the outer iteration met its stopping rule within its `maxIter` iterations
- `intercept`: the unpenalized intercept, zero unless the solver was called
  with `intercept = true`
"""
struct LassoSolution{T, S}
  x::SparseIterate{T}
  residuals::Vector{T}
  penalty::S
  σ::Union{T, Nothing}
  iterations::Int
  converged::Bool
  intercept::T

  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::ProxL1{T}, σ::T,
                      iterations::Integer, converged::Bool, intercept::T) where {T, S} =
    new(x, residuals, penalty, σ, iterations, converged, intercept)
end

# the penalty loadings of standardized columns: the standard deviations of the
# columns, centered when f fits an intercept
_loadings(f::CoordinateDifferentiableFunction) = [coordinateScale(f, k) for k in 1:numCoordinates(f)]


######################################################################
#
#   Lasso Interface
#
######################################################################

"""
    lasso(X, y, λ, options=CDOptions(); intercept=false, standardizeX=false)
    lasso(X, y, λ, ω, options=CDOptions(); intercept=false)

Solve the lasso

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent. The intercept `β₀` is not penalized, and it is zero
unless `intercept` is true. In the first form the loadings are `ωⱼ = 1`, or
with `standardizeX` the standard deviations `sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the
columns, with `x̄ⱼ` the column mean when `intercept` is true and zero
otherwise; the coefficients are on the scale of the data either way. Returns a
[`LassoSolution`](@ref) whose `σ` is the sample standard deviation of the
residuals.
"""
function lasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  standardizeX::Bool=false) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDLeastSquaresLoss(y, X; intercept)
  g = standardizeX ? ProxL1(λ, _loadings(f)) : ProxL1(λ)
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged, _intercept(f, x))
end

function lasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDLeastSquaresLoss(y, X; intercept)
  g = ProxL1(λ, ω)
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged, _intercept(f, x))
end

function lasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), options; kwargs...)
end

function lasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), convert(Array{T}, ω), options; kwargs...)
end

######################################################################
#
#   Sqrt-Lasso Interface
#
######################################################################

"""
    sqrtLasso(X, y, λ, options=CDOptions(); intercept=false, standardizeX=true)
    sqrtLasso(X, y, λ, ω, options=CDOptions(); intercept=false)

Solve the square-root lasso of Belloni, Chernozhukov and Wang (2011)

    minimize  ‖y - β₀ - Xβ‖ + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent, with an unpenalized intercept `β₀` that is zero unless
`intercept` is true. In the first form the loadings are the standard deviations
`ωⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the columns when `standardizeX` is true
(centered only with an intercept) and `ωⱼ = 1` otherwise. The optimal `λ`
does not depend on the noise level. Returns a [`LassoSolution`](@ref) whose
`σ` is the sample standard deviation of the residuals.
"""
function sqrtLasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  standardizeX::Bool=true) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDSqrtLassoLoss(y, X; intercept)
  g = standardizeX ? ProxL1(λ, _loadings(f)) : ProxL1(λ)
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged, _intercept(f, x))
end

function sqrtLasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  f = CDSqrtLassoLoss(y, X; intercept)
  g = ProxL1(λ, ω)
  info = CDInfo()
  coordinateDescent!(x, f, g, options, info)

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), info.iterations, info.converged, _intercept(f, x))
end

function sqrtLasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), options; kwargs...)
end

function sqrtLasso(X::StridedMatrix{<:Real}, y::StridedVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), convert(Array{T}, ω), options; kwargs...)
end


######################################################################
#
#   Scaled Lasso Interface
#
######################################################################

"""
    scaledLasso!(x, X, y, λ, ω, options=IterLassoOptions(); intercept=false)
    scaledLasso(X, y, λ, ω, options=IterLassoOptions(); intercept=false)

Scaled lasso of Sun and Zhang (2012), which estimates the coefficients and the
noise level jointly. It alternates between the lasso

    β̂ = argmin  ‖y - β₀ - Xβ‖² / (2n) + λ σ̂ ∑ⱼ ωⱼ |βⱼ|

and the update `σ̂ = ‖y - β̂₀ - Xβ̂‖ / √n` until the relative change in `σ̂` is
below `options.optTol`, or warns after `options.maxIter` iterations. The
intercept `β₀` is not penalized, and it is zero unless `intercept` is true.

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
  options::IterLassoOptions=IterLassoOptions();
  intercept::Bool=false
  )  where {T<:AbstractFloat}

  n = size(X, 1)
  f = CDLeastSquaresLoss(y, X; intercept)

  # initialize σ
  if options.initProcedure == :Screening
    σ = _findInitSigma!(X, y, options.sinit, f.r; intercept)
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

  LassoSolution{T, typeof(g)}(x, f.r, g, sqrt( sum(abs2, f.r) / n ), iterations, σconverged && info.converged,
                              _intercept(f, x))
end

scaledLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ::Real,
  ω::AbstractVector{<:Real},
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} =
  scaledLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ),
               convert(AbstractVector{T}, ω), options; kwargs...)

"""
    scaledLasso(X, y, λ, ω, options=IterLassoOptions())

Same as [`scaledLasso!`](@ref), starting from a zero iterate.
"""
scaledLasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::AbstractVector{T},
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} = scaledLasso!(SparseIterate(T, size(X, 2)), X, y, λ, ω, options; kwargs...)

function scaledLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  ω::AbstractVector{<:Real}, options::IterLassoOptions=IterLassoOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  scaledLasso(Xf, yf, T(λ), convert(AbstractVector{T}, ω), options; kwargs...)
end


######################################################################
#
#   Feasible Lasso Interface
#
######################################################################

"""
    feasibleLasso!(x, X, y, λ0, options=IterLassoOptions(); intercept=false)

Heteroscedasticity-robust lasso of Belloni, Chen, Chernozhukov and Hansen
(2012), which uses data-driven penalty loadings. It alternates between the
lasso

    β̂ = argmin  ‖y - β₀ - Xβ‖² / (2n) + λ0 ∑ⱼ Γⱼ |βⱼ|

and the update of the loadings `Γⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² rᵢ² / n)` from the
residuals `r = y - β̂₀ - Xβ̂`, until the relative change in `Γ` is below
`options.optTol`, or warns after `options.maxIter` iterations. The intercept
`β₀` is not penalized; it and the column means `x̄ⱼ` are zero unless
`intercept` is true.

The initial residuals are chosen by `options.initProcedure` (see
[`IterLassoOptions`](@ref)); with `:InitStd` they come from a lasso with
penalty `λ0 σinit ∑ⱼ sⱼ |βⱼ|`, `sⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)`. Coordinate descent
starts from and overwrites `x`. Returns a [`LassoSolution`](@ref) whose
`penalty.λ` holds the loadings and whose `σ` is the sample standard deviation of
the residuals.
"""
function feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ0::T,
  options::IterLassoOptions=IterLassoOptions();
  intercept::Bool=false
  )  where {T<:AbstractFloat}

  p = size(X, 2)
  f = CDLeastSquaresLoss(y, X; intercept)
  Γ = Array{T}(undef, p)              # stores loadings
  Γold = Array{T}(undef, p)

  info = CDInfo()
  iterations = 0

  # initialize residuals
  if options.initProcedure == :Screening
    _findInitResiduals!(X, y, options.sinit, f.r; intercept)
  elseif options.initProcedure == :InitStd
    σ = options.σinit
    Γ .= _loadings(f)
    coordinateDescent!(x, f, ProxL1(λ0*σ, Γ), options.optionsCD, info)
    iterations += info.iterations
  elseif options.initProcedure == :WarmStart
    initialize!(f, x)
  else
    throw(ArgumentError("Incorrect initialization Symbol"))
  end
  _getLoadings!(Γ, X, f.r, f.xbar)

  # g holds a reference to Γ, so updating Γ in place updates the penalty
  g = ProxL1(λ0, Γ)
  Γconverged, change = false, NaN
  for iter=1:options.maxIter
    copyto!(Γold, Γ)

    coordinateDescent!(x, f, g, options.optionsCD, info)
    iterations += info.iterations
    _getLoadings!(Γ, X, f.r, f.xbar)

    change = maximum(abs.(Γold  - Γ)) / maximum(Γ)
    if change < options.optTol
      Γconverged = true
      break
    end
  end
  Γconverged || @warn "feasibleLasso! did not converge in maxIter = $(options.maxIter) iterations (relative change of the penalty loadings in the last iteration $(round(change; sigdigits = 3)) > optTol = $(options.optTol)). Increase maxIter in IterLassoOptions."

  LassoSolution{T, typeof(g)}(x, f.r, g, std(f.r), iterations, Γconverged && info.converged, _intercept(f, x))
end

feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ0::Real,
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} =
  feasibleLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ0), options; kwargs...)

######################################################################
#
#   Lasso Path Interface
#
######################################################################

"""
    LassoPath(X, y, λpath, options=CDOptions(); max_hat_s=Inf, standardizeX=true, intercept=false, screening=true)
    LassoPath(X, y, options=CDOptions(); nλ=100, λminratio, kwargs...)

Lasso solutions `β0path[k]` and `βpath[k]` of

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λpath[k] ∑ⱼ ωⱼ |βⱼ|

along a decreasing sequence `λpath`, each one warm started from the previous
solution. The intercept `β₀` is not penalized, and it is zero unless
`intercept` is true. The loadings are the standard deviations
`ωⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the columns when `standardizeX` is true
(centered only with an intercept) and `ωⱼ = 1` otherwise.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`, so that rounding
cannot make the first solution nonzero.

With `screening`, each λ is solved first over the coordinates kept by the
sequential strong rule of Tibshirani et al. (2012), and the optimality
conditions of the others are checked afterwards; coordinates that violate them
are added and the problem solved again. `kktChecks[k]` is the number of checks
at `λpath[k]`: one when the rule kept every coordinate it should have, zero
without screening. Screening needs warm starts, so it is off when
`options.warmStart` is false.

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
  β0path::Vector{T}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kktChecks::Vector{Int}
  intercept::Bool
end

"""
    refitLassoPath(path::LassoPath, X, y)

Least squares refits on the supports along a [`LassoPath`](@ref). Returns a
`Dict` that maps each distinct support `S` (a vector of column indices) to
`X[:, S] \\ y`, or, when the path has an intercept, to `[1 X[:, S]] \\ y`,
whose first entry is the intercept.
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
    out[S] = path.intercept ? [ones(T, length(Y)) X[:, S]] \ Y : X[:, S] \ Y
  end
  out
end


function LassoPath(
  X::StridedMatrix{T},
  Y::StridedVector{T},
  λpath::Vector{T},
  options=CDOptions();
  max_hat_s=Inf, standardizeX::Bool=true, intercept::Bool=false,
  screening::Bool=true) where {T<:AbstractFloat}

  p = size(X, 2)
  x = SparseIterate(T, p)
  f = CDLeastSquaresLoss(Y, X; intercept)
  stdX = standardizeX ? _loadings(f) : ones(T, p)

  screen = screening && options.warmStart
  if screen
    initialize!(f, x)
    coef_iterator = options.randomize ? RandomIterator(x) : OrderedIterator(x)
    grad = Vector{T}(undef, p)
    fullGradient!(grad, f, x)
    inS = falses(p)
    λprev = _findLambdaMax(x, f, ProxL1(one(T), stdX))
  end

  βpath = Vector{SparseVector{T, Int}}()
  β0path = T[]
  iterations, converged, kktChecks = Int[], Bool[], Int[]
  info = CDInfo()
  for λ in λpath
    g = ProxL1(λ, stdX)
    if screen
      initialize!(f, x)       # no drift of the residuals along the path
      checks = _screenedSolve!(x, f, g, λprev, grad, coef_iterator, options, info, inS)
      _warn_not_converged(info, options)
      λprev = λ
    else
      coordinateDescent!(x, f, g, options, info)
      checks = 0
    end
    push!(βpath, _sparsevec(x))
    push!(β0path, _intercept(f, x))
    push!(iterations, info.iterations)
    push!(converged, info.converged)
    push!(kktChecks, checks)
    if nnz(x) > max_hat_s
      break
    end
  end

  LassoPath{T}(λpath[1:length(βpath)], βpath, β0path, iterations, converged, kktChecks, intercept)
end

function LassoPath(
  X::StridedMatrix{T},
  Y::StridedVector{T},
  options::CDOptions=CDOptions();
  nλ::Integer=100,
  λminratio::Real=(size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
  standardizeX::Bool=true,
  intercept::Bool=false,
  kwargs...) where {T<:AbstractFloat}

  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  p = size(X, 2)
  f = CDLeastSquaresLoss(Y, X; intercept)
  ω = standardizeX ? _loadings(f) : ones(T, p)
  x = SparseIterate(T, p)
  initialize!(f, x)
  λmax = _findLambdaMax(x, f, ProxL1(one(T), ω)) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  LassoPath(X, Y, λpath, options; standardizeX, intercept, kwargs...)
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

function LassoPath(
  X::StridedMatrix{<:Real},
  Y::StridedVector{<:Real},
  options::CDOptions=CDOptions();
  kwargs...)
  T, Xf, Yf = _float_data(X, Y)
  LassoPath(Xf, Yf, options; kwargs...)
end
