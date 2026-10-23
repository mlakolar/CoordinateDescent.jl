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
- `penalty::NormL1`: the penalty at which `x` was computed
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

  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::NormL1{T}, σ::T,
                      iterations::Integer, converged::Bool, intercept::T) where {T, S} =
    new(x, residuals, penalty, σ, iterations, converged, intercept)
end

# the penalty loadings of standardized columns: the standard deviations
# sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n) of the columns, centered at their means x̄ⱼ with an
# intercept and at zero otherwise
function _loadings(X::AbstractMatrix{T}, intercept::Bool) where {T}
  n, p = size(X)
  xbar = intercept ? _column_means(X) : zeros(T, p)
  ω = Vector{T}(undef, p)
  @inbounds for k in 1:p
    c = xbar[k]
    a = zero(T)
    @simd for i in 1:n
      a += (X[i, k] - c) * (X[i, k] - c)
    end
    ω[k] = sqrt(a / n)
  end
  ω
end

# the solution of a lasso fit by the model m at the penalty g
function _lassoSolution(x::SparseIterate{T}, m, g, r::CDResults) where {T}
  res = residuals(m)
  LassoSolution{T, typeof(g)}(x, res, g, std(res), r.passes, r.converged, ProximalSolvers.intercept(m, x))
end


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
  m = CDLeastSquaresLoss(y, X; intercept)
  g = standardizeX ? NormL1(λ, _loadings(X, intercept)) : NormL1(λ)
  _lassoSolution(x, m, g, _solve!(x, m, g, options))
end

function lasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  m = CDLeastSquaresLoss(y, X; intercept)
  g = NormL1(λ, ω)
  _lassoSolution(x, m, g, _solve!(x, m, g, options))
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
  m = CDSqrtLassoLoss(y, X; intercept)
  g = standardizeX ? NormL1(λ, _loadings(X, intercept)) : NormL1(λ)
  _lassoSolution(x, m, g, _solve!(x, m, g, options))
end

function sqrtLasso(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false) where {T<:AbstractFloat}

  x = SparseIterate(T, size(X, 2))
  m = CDSqrtLassoLoss(y, X; intercept)
  g = NormL1(λ, ω)
  _lassoSolution(x, m, g, _solve!(x, m, g, options))
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
    scaledLasso!(x, f::CDSqrtQuadraticLoss, λ, ω, options=CDOptions())

Scaled lasso of Sun and Zhang (2012), which estimates the coefficients and the
noise level jointly:

    minimize  ‖y - β₀ - Xβ‖² / (2nσ) + σ/2 + λ ∑ⱼ ωⱼ |βⱼ|   over β and σ > 0.

The intercept `β₀` is not penalized, and it is zero unless `intercept` is true.
The minimum over `σ` is at `σ̂ = ‖y - β₀ - Xβ‖ / √n`, so `β̂` solves the
square-root lasso of Belloni, Chernozhukov and Wang (2011),

    minimize  ‖y - β₀ - Xβ‖ / √n + λ ∑ⱼ ωⱼ |βⱼ|,

which one run of coordinate descent solves (on [`CDSqrtLassoLoss`](@ref), with
the penalty level `√n λ`), with the options `options.optionsCD`; the other
fields of [`IterLassoOptions`](@ref) are not used. Coordinate descent starts
from and overwrites `x` (unless `options.optionsCD.warmStart` is false).
Returns a [`LassoSolution`](@ref) whose `σ` is `σ̂`, and whose penalty is that of
the lasso that `β̂` solves, `NormL1(λσ̂, ω)`.

The third form solves the same problem from sufficient statistics: with the
loss [`CDSqrtQuadraticLoss`](@ref)`(A, b, c)`, `A = X'X/n`, `b = -X'y/n` and
`c = y'y/n`, it needs neither `X` nor `y`, and a coordinate update costs `O(p)`
instead of `O(n)`. With `A = S`, `b = -S[:, j]`, `c = S[j, j]` and `ω[j] = Inf`
it is the scaled lasso of variable `j` on the others from their covariance
matrix `S`, as in the precision matrix estimator of Sun and Zhang (2013); for
all the variables, check `S` once and build the losses with `check = false`.
Its solution has no residuals (an empty vector) and no intercept.

Before version 0.6.1, `scaledLasso!` alternated lasso fits with updates of `σ̂`
until the relative change of `σ̂` was below `options.optTol` (by default 1e-2),
which stopped short of the solution and solved a lasso at every iteration.
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
  m = CDSqrtLassoLoss(y, X; intercept)
  r = _solve!(x, m, NormL1(sqrt(T(n)) * λ, ω), options.optionsCD)
  res = residuals(m)
  σ = norm(res) / sqrt(T(n))
  g = NormL1(λ * σ, ω)
  LassoSolution{T, typeof(g)}(x, res, g, σ, r.passes, r.converged, ProximalSolvers.intercept(m, x))
end

# reweight! with the options of the iterative procedures, warm starting each
# lasso fit from the previous one; with maxIter = 0 there is no fit
function _reweight!(x, m, g, update, options::IterLassoOptions)
  if options.maxIter == 0
    ProximalSolvers.initialize!(m, x)
    return ReweightResults(x, g, g, 0, false, NaN, 0, true)
  end
  reweight!(x, m, g, update, _solver(options.optionsCD); maxiter = options.maxIter, tol = options.optTol)
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

function scaledLasso!(
  x::SparseIterate{T},
  f::SqrtQuadModel{T},
  λ::T,
  ω::AbstractVector{T},
  options::CDOptions=CDOptions()) where {T<:AbstractFloat}

  r = _solve!(x, f, NormL1(λ, ω), options)
  σ = ProximalSolvers.loss_value(f, x)
  g = NormL1(λ * σ, ω)
  LassoSolution{T, typeof(g)}(x, T[], g, σ, r.passes, r.converged, zero(T))
end

scaledLasso!(x::SparseIterate{T}, f::SqrtQuadModel{T}, λ::Real, ω::AbstractVector{<:Real},
             options::CDOptions=CDOptions()) where {T<:AbstractFloat} =
  scaledLasso!(x, f, T(λ), convert(AbstractVector{T}, ω), options)

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
`options.optTol` (by `ProximalSolvers.reweight!`), or warns after
`options.maxIter` iterations. The intercept `β₀` is not penalized; it and the
column means `x̄ⱼ` are zero unless `intercept` is true.

The initial residuals are chosen by `options.initProcedure` (see
[`IterLassoOptions`](@ref)); with `:InitStd` they come from a lasso with
penalty `λ0 σinit ∑ⱼ sⱼ |βⱼ|`, `sⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)`. Coordinate descent
starts from and overwrites `x`, and each lasso fit is warm started from the
previous one. Returns a [`LassoSolution`](@ref) whose `penalty.w` holds the
loadings of the last fit and whose `σ` is the sample standard deviation of the
residuals.
"""
function feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ0::T,
  options::IterLassoOptions=IterLassoOptions();
  intercept::Bool=false
  )  where {T<:AbstractFloat}

  n, p = size(X)
  m = CDLeastSquaresLoss(y, X; intercept)
  xbar = intercept ? _column_means(X) : zeros(T, p)
  passes, cd_converged = 0, true

  # initial residuals; IterLassoOptions allows only these three procedures
  if options.initProcedure == :Screening
    r0 = _findInitResiduals!(X, y, options.sinit, Vector{T}(undef, n); intercept)
  elseif options.initProcedure == :InitStd
    σ = T(options.σinit)
    r = _solve!(x, m, NormL1(λ0 * σ, _loadings(X, intercept)), options.optionsCD)
    passes, cd_converged = r.passes, r.converged
    r0 = residuals(m)
  else    # :WarmStart
    ProximalSolvers.initialize!(m, x)
    r0 = residuals(m)
  end

  # alternate the lasso with loadings Γ with the update of Γ from its residuals
  Γ = _getLoadings!(Vector{T}(undef, p), X, r0, xbar)
  function update(g, x, m)
    Γnew = _getLoadings!(Vector{T}(undef, p), X, residuals(m), xbar)
    NormL1(λ0, Γnew), maximum(abs.(g.w .- Γnew)) / maximum(Γnew)
  end
  res = _reweight!(x, m, NormL1(λ0, Γ), update, options)
  res.converged || @warn "feasibleLasso! did not converge in maxIter = $(options.maxIter) iterations (relative change of the penalty loadings in the last iteration $(round(res.change; sigdigits = 3)) > optTol = $(options.optTol)). Increase maxIter in IterLassoOptions."

  r = residuals(m)
  LassoSolution{T, typeof(res.penalty)}(x, r, res.penalty, std(r), passes + res.passes,
                                        res.converged && res.cd_converged && cd_converged,
                                        ProximalSolvers.intercept(m, x))
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
solution, by `ProximalSolvers.solve_path`. The intercept `β₀` is not penalized, and it is zero unless
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
without screening. When `options.warmStart` is false, each value is solved
from zero along its own path of `options.numSteps` levels, without screening.

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

  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  p = size(X, 2)
  x = SparseIterate(T, p)
  m = CDLeastSquaresLoss(Y, X; intercept)
  ω = standardizeX ? _loadings(X, intercept) : ones(T, p)
  options.warmStart || return _coldLassoPath(x, m, ω, λpath, options, max_hat_s, intercept)

  β0path = T[]
  path = solve_path(m, NormL1(one(T), ω), λpath, _solver(options); x, screening, maxnnz = max_hat_s,
                    callback = (k, λ, x, m) -> (push!(β0path, ProximalSolvers.intercept(m, x)); false))
  LassoPath{T}(path.λ, path.solutions, β0path, path.passes, path.converged, path.kkt_checks, intercept)
end

# the path with warmStart = false: each level solved from zero along its own
# path of options.numSteps levels, without screening
function _coldLassoPath(x::SparseIterate{T}, m, ω, λpath::Vector{T}, options::CDOptions, max_hat_s,
                        intercept::Bool) where {T}
  βpath = SparseVector{T, Int}[]
  β0path, iterations, converged = T[], Int[], Bool[]
  for λ in λpath
    r = _solve!(x, m, NormL1(λ, ω), options)
    push!(βpath, _sparsevec(x))
    push!(β0path, ProximalSolvers.intercept(m, x))
    push!(iterations, r.passes)
    push!(converged, r.converged)
    nnz(x) > max_hat_s && break
  end
  k = length(βpath)
  LassoPath{T}(λpath[1:k], βpath, β0path, iterations, converged, zeros(Int, k), intercept)
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
  m = CDLeastSquaresLoss(Y, X; intercept)
  ω = standardizeX ? _loadings(X, intercept) : ones(T, p)
  λmax = lambda_max(m, NormL1(one(T), ω), SparseIterate(T, p)) * (1 + sqrt(eps(T)))
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
