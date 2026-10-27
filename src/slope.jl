# SLOPE uses the sorted penalty of ProximalBase and the accelerated proximal
# gradient path engine of ProximalSolvers. Data preparation, calibration and
# statistical result objects remain here. All penalties here use loss / n.

"""
    slope_bh(p; q=0.1, sigma=1, n=1)

The Benjamini–Hochberg sequence `sigma / sqrt(n) * Φ⁻¹(1 - q*j/(2p))`,
`j = 1:p`, for [`slope`](@ref)'s loss `‖y - Xβ‖²/(2n)`. Under independent
Gaussian errors of known standard deviation `sigma` and `X'X/n = I`, SLOPE
with this sequence controls FDR at most `q`. Standardizing correlated columns
or estimating `sigma` does not establish that guarantee.
"""
function slope_bh(p::Integer; q::Real=0.1, sigma::Real=1, n::Integer=1)
  p > 0 && n > 0 || throw(ArgumentError("p and n must be positive"))
  isfinite(q) && 0 < q < 1 || throw(ArgumentError("q must be in (0, 1)"))
  isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
  [sigma / sqrt(n) * sqrt(2) * erfcinv(q * j / p) for j in 1:p]
end

"""
    slope_gaussian(p; n, q=0.1, sigma=1, intercept=false)

Gaussian-adjusted SLOPE weights for loss `‖y - β₀ - Xβ‖²/(2n)`.
With `m = n - Int(intercept)` and `b = slope_bh(p; q)`, compute
`a[1] = b[1]` and `a[i] = b[i]*sqrt(1 + sum(a[1:i-1].^2)/(m-i))`.
At the first nondecrease or nonpositive denominator, keep the preceding
weight for the remaining entries. Return `sigma/sqrt(n) * a`.

This is the Gaussian-design approximation of Bogdan et al. (2015), equations
(26)–(27), with an explicit first-nondecrease truncation and an intercept
dimension adjustment. It is not a finite-sample FDR guarantee for correlated
designs, estimated noise, or selected path levels. The design should have
independent Gaussian columns scaled to mean square one. `m` must be positive;
if `p > 1`, require `m ≥ 2`. Wide designs are supported by the documented
flat-tail rule, without evaluating the recurrence at a pole.

The denominator is `m-i` in one-based indexing. SLOPE.jl 1.3.1 instead uses
`max(1, n-i-1)`; its Gaussian weights are a distinct convention.
"""
slope_gaussian(p::Integer; n::Integer, q::Real=0.1, sigma::Real=1, intercept::Bool=false) =
  first(_slope_gaussian(p, n, q, sigma, intercept))

function _slope_gaussian(p, n, q, sigma, intercept)
  # Form dimensionless weights before sigma scaling, including when sigma=0.
  a = slope_bh(p; q, sigma=1)
  isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
  m = n - Int(intercept)
  m > 0 && (p == 1 || m >= 2) ||
    throw(ArgumentError("Gaussian calibration needs n-intercept ≥ 2 when p > 1 (≥ 1 when p = 1)"))
  all(isfinite, a) || throw(ArgumentError("q is too small for finite Gaussian weights at this dimension"))
  total = zero(eltype(a))
  flat_from, reason = nothing, :none
  for i in 2:p
    total += abs2(a[i-1])
    denominator = m - i
    if denominator <= 0
      a[i:end] .= a[i-1]
      flat_from, reason = i, :dimension_boundary
      break
    end
    next = a[i] * sqrt(1 + total / denominator)
    if !isfinite(next) || next >= a[i-1]
      a[i:end] .= a[i-1]
      flat_from, reason = i, :first_nondecrease
      break
    end
    a[i] = next
  end
  a .*= sigma / sqrt(n)
  all(isfinite, a) || throw(ArgumentError("sigma gives nonfinite Gaussian weights"))
  a, flat_from, reason
end

function _slope_calibration(p, n, ::Type{T}; lambda, calibration, q, sigma, intercept, standardize_x) where {T}
  calibration in (:bh, :gaussian, :user) || throw(ArgumentError("calibration must be :bh, :gaussian, or :user"))
  if lambda !== nothing
    info = (method=:user, requested=calibration, q=nothing, sigma=nothing,
            noise=:user_weights, n, effective_n=n-Int(intercept),
            flat_from=nothing, flatten_reason=:none, standardize_x)
    return _slope_lambda(lambda, p, T), info
  end
  calibration == :user && throw(ArgumentError("calibration=:user requires explicit lambda weights"))
  if calibration == :bh
    weights, flat_from, reason = slope_bh(p; q, sigma, n), nothing, :none
  else
    weights, flat_from, reason = _slope_gaussian(p, n, q, sigma, intercept)
  end
  info = (method=calibration, requested=calibration, q, sigma, noise=:supplied_sigma,
          n, effective_n=n-Int(intercept), flat_from, flatten_reason=reason, standardize_x)
  _slope_lambda(weights, p, T), info
end

# A sparse standardized design, centered by a rank-one correction. Scaling
# touches stored entries once; centering never fills the implicit zeros.
struct _SLOPESparseDesign{T,I,C} <: AbstractMatrix{T}
  X::SparseMatrixCSC{T,I}
  center::C
end
Base.size(Z::_SLOPESparseDesign) = size(Z.X)
Base.getindex(Z::_SLOPESparseDesign, i::Int, j::Int) =
  Z.X[i,j] - (Z.center === nothing ? zero(eltype(Z)) : Z.center[j])
Base.getindex(Z::_SLOPESparseDesign, ::Colon, E::AbstractVector{<:Integer}) =
  _SLOPESparseDesign(Z.X[:,E], Z.center === nothing ? nothing : Z.center[E])

function LinearAlgebra.mul!(out::AbstractVector, Z::_SLOPESparseDesign,
                            b::AbstractVector, α::Number, δ::Number)
  mul!(out, Z.X, b, α, δ)
  if Z.center !== nothing
    shift = α * dot(Z.center, b)
    out .-= shift
  end
  out
end
LinearAlgebra.mul!(out::AbstractVector, Z::_SLOPESparseDesign, b::AbstractVector) =
  mul!(out, Z, b, true, false)

function LinearAlgebra.mul!(out::AbstractVector,
                            A::Union{Transpose{<:Any,<:_SLOPESparseDesign},Adjoint{<:Any,<:_SLOPESparseDesign}},
                            b::AbstractVector, α::Number, δ::Number)
  Z = parent(A)
  mul!(out, transpose(Z.X), b, α, δ)
  if Z.center !== nothing
    shift = α * sum(b)
    out .-= shift .* Z.center
  end
  out
end
LinearAlgebra.mul!(out::AbstractVector,
                   A::Union{Transpose{<:Any,<:_SLOPESparseDesign},Adjoint{<:Any,<:_SLOPESparseDesign}},
                   b::AbstractVector) = mul!(out, A, b, true, false)

# Validation and centering are shared by the fit and the path. Keep the
# existing explicit transformation and BLAS products for dense inputs.
function _slope_data(X, y, intercept, standardize_x)
  n, p = size(X)
  n > 0 && p > 0 || throw(ArgumentError("X must have positive dimensions"))
  length(y) == n || throw(DimensionMismatch("X and y have different observation counts"))
  T = float(promote_type(eltype(X), eltype(y)))
  Z, v = Matrix{T}(X), Vector{T}(y)
  all(isfinite, Z) && all(isfinite, v) || throw(ArgumentError("X and y must be finite"))
  xb = intercept ? _column_means(Z) : zeros(T, p)
  yb = intercept ? mean(v) : zero(T)
  Z .-= transpose(xb)
  v .-= yb
  scales = standardize_x ? vec(sqrt.(sum(abs2, Z; dims=1) ./ n)) : ones(T, p)
  # A zero column has coefficient zero; use unit scale to avoid dividing by zero.
  replace!(scales, zero(T) => one(T))
  Z ./= transpose(scales)
  Z, v, xb, yb, scales
end

function _slope_data(X::SparseMatrixCSC, y, intercept, standardize_x)
  n, p = size(X)
  n > 0 && p > 0 || throw(ArgumentError("X must have positive dimensions"))
  length(y) == n || throw(DimensionMismatch("X and y have different observation counts"))
  T = float(promote_type(eltype(X), eltype(y)))
  A = copy(SparseMatrixCSC{T,eltype(rowvals(X))}(X))
  v = Vector{T}(y)
  all(isfinite, nonzeros(A)) && all(isfinite, v) || throw(ArgumentError("X and y must be finite"))
  xb = intercept ? _column_means(A) : zeros(T, p)
  yb = intercept ? mean(v) : zero(T)
  v .-= yb
  scales = standardize_x ? _loadings(A, intercept) : ones(T, p)
  replace!(scales, zero(T) => one(T))
  center = intercept ? xb ./ scales : nothing
  vals = nonzeros(A)
  for j in 1:p
    r = nzrange(A, j)
    if intercept && length(r) == n
      # A fully stored column can be centered before scaling without filling
      # implicit zeros. This avoids cancelling large products for offset columns
      # and makes constant columns exactly zero in both forward and adjoint maps.
      for k in r
        vals[k] = (vals[k] - xb[j]) / scales[j]
      end
      center[j] = zero(T)
    elseif standardize_x
      for k in r
        vals[k] /= scales[j]
      end
    end
  end
  Z = _SLOPESparseDesign(A, center)
  Z, v, xb, yb, scales
end

function _slope_lambda(lambda, p, ::Type{T}) where {T}
  length(lambda) == p || throw(DimensionMismatch("lambda must have one entry per predictor"))
  λ = Vector{T}(lambda)
  all(isfinite, λ) && all(>=(0), λ) && issorted(λ; rev=true) ||
    throw(ArgumentError("lambda must be finite, nonnegative and nonincreasing"))
  λ
end

"""
    SLOPESolution

Result of [`slope`](@ref). `x` contains the coefficients on the input scale;
`intercept`, `residuals`, `lambda`, `iterations`, `converged` and `kkt_residual`
describe the fit. The convergence flag requires a full proximal KKT check.
Supports `coef`, `predict`, `residuals`, `nobs`, and `deviance` from StatsAPI.
"""
struct SLOPESolution{T}
  x::Vector{T}
  intercept::T
  hasintercept::Bool
  residuals::Vector{T}
  lambda::Vector{T}
  iterations::Int
  converged::Bool
  kkt_residual::T
  calibration::NamedTuple
end

# Preserve the pre-calibration positional result constructor.
SLOPESolution(x, b0, hasintercept, residuals, lambda, iterations, converged, residual) =
  SLOPESolution(x, b0, hasintercept, residuals, lambda, iterations, converged, residual, (method=:unspecified,))
SLOPESolution{T}(x, b0, hasintercept, residuals, lambda, iterations, converged, residual) where {T} =
  SLOPESolution{T}(x, b0, hasintercept, residuals, lambda, iterations, converged, residual, (method=:unspecified,))

function _slope_options(max_iter, tol)
  max_iter >= 2 || throw(ArgumentError("max_iter must be at least 2"))
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
end

"""
    slope(X, y, lambda; intercept=false, standardize_x=false, max_iter=10000, tol=1e-7)
    slope(X, y; lambda=nothing, calibration=:bh, q=0.1, sigma=1, kwargs...)

Minimize `‖y - β₀ - Xβ‖²/(2n) + ∑ⱼ lambda[j]*|β|₍ⱼ₎` by accelerated
proximal gradient. `lambda` must be nonincreasing and nonnegative. The second
form uses [`slope_bh`](@ref) by default; `calibration=:gaussian` selects
[`slope_gaussian`](@ref). Explicit `lambda` weights always take precedence
over either automatic rule; `calibration=:user` requires such weights.
`sol.calibration` records the rule, supplied sigma convention and truncation.
With `standardize_x`, the
penalty applies to the coefficients of columns scaled to mean square one;
returned coefficients are on the input scale. An intercept centers X and y.
The design is copied to floating-point storage. `SparseMatrixCSC` inputs stay
sparse, including with centering and standardization. Failure of the full
optimality check gives `converged=false` and a warning.
"""
function slope(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
               lambda=nothing, calibration::Symbol=:bh, q::Real=0.1, sigma::Real=1,
               intercept::Bool=false, standardize_x::Bool=false, max_iter::Integer=10000, tol::Real=1e-7)
  _slope_options(max_iter, tol)
  Z, v, xb, yb, scales = _slope_data(X, y, intercept, standardize_x)
  β = zeros(eltype(Z), size(Z, 2))
  λ, info = _slope_calibration(length(β), length(v), eltype(β); lambda, calibration, q, sigma, intercept, standardize_x)
  f = (one(eltype(Z)) / length(v)) * LeastSquaresLoss(v, Z)
  r = ProximalSolvers.solve!(β, f, ProximalBase.SLOPE(λ),
    ProximalSolvers.SLOPEAPG(; maxiter=max_iter, tol))
  it, ok, residual = r.iterations, r.converged, r.kkt_residual
  ok || @warn "SLOPE did not meet its full optimality tolerance" residual iterations=it
  β ./= scales
  β0 = yb - dot(xb, β)
  SLOPESolution(β, β0, intercept, Vector(y .- β0 .- X * β), λ, it, ok, residual, info)
end

slope(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::AbstractVector{<:Real}; kwargs...) =
  slope(X, y; lambda, kwargs...)

"""
    SLOPEPath(X, y, factors=nothing; lambda=nothing, calibration=:bh, q=0.1, sigma=1,
              nλ=50, λmin_ratio=0.01, screening=true, intercept=false,
              standardize_x=false, max_iter=10000, tol=1e-7)

Warm-started SLOPE fits with penalties `factor * lambda`. `lambda` defaults
to [`slope_bh`](@ref); `calibration=:gaussian` selects [`slope_gaussian`](@ref).
Explicit weights always take precedence. `path.calibration` records the actual
rule and any truncation. Supply positive, nonincreasing `factors`, or a geometric
path from the smallest factor with a zero fit to `λmin_ratio` times it.
`screening` uses the strong rule of Larsson, Bogdan and Wallin (2020), with
full proximal KKT checks and repairs. `screening=false` fits every column.

`λpath` holds the factors, `lambda` the base sequence, `βpath` the coefficient
vectors, and `β0path` the intercepts. Per-level diagnostics are `converged`,
`iterations`, `kkt_residuals`, `working_sizes`, and `kkt_repairs`.
`coef(path)` and `predict(path, X)` have a column per level.
"""
struct SLOPEPath{T}
  λpath::Vector{T}
  lambda::Vector{T}
  βpath::Vector{Vector{T}}
  β0path::Vector{T}
  intercept::Bool
  nobs::Int
  converged::BitVector
  iterations::Vector{Int}
  kkt_residuals::Vector{T}
  working_sizes::Vector{Int}
  kkt_repairs::Vector{Int}
  calibration::NamedTuple
end

SLOPEPath(a, lambda, bs, b0, intercept, nobs, converged, iterations, residuals, sizes, repairs) =
  SLOPEPath(a, lambda, bs, b0, intercept, nobs, converged, iterations, residuals, sizes, repairs, (method=:unspecified,))
SLOPEPath{T}(a, lambda, bs, b0, intercept, nobs, converged, iterations, residuals, sizes, repairs) where {T} =
  SLOPEPath{T}(a, lambda, bs, b0, intercept, nobs, converged, iterations, residuals, sizes, repairs, (method=:unspecified,))

function SLOPEPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, factors=nothing;
                   lambda=nothing, calibration::Symbol=:bh, q::Real=0.1, sigma::Real=1, nλ::Integer=50, λmin_ratio::Real=0.01,
                   screening::Bool=true, intercept::Bool=false, standardize_x::Bool=false,
                   max_iter::Integer=10000, tol::Real=1e-7)
  _slope_options(max_iter, tol)
  Z, v, xb, yb, scales = _slope_data(X, y, intercept, standardize_x)
  T, p = eltype(Z), size(Z, 2)
  λ, info = _slope_calibration(p, length(y), T; lambda, calibration, q, sigma, intercept, standardize_x)
  β = zeros(T, p)
  f = (one(T) / length(v)) * LeastSquaresLoss(v, Z)
  g = ProximalBase.SLOPE(λ)
  if factors === nothing
    nλ > 0 && isfinite(λmin_ratio) && 0 < λmin_ratio <= 1 ||
      throw(ArgumentError("nλ must be positive and λmin_ratio in (0, 1]"))
    λ[1] > 0 || throw(ArgumentError("automatic paths need a nonzero base sequence"))
    amax = ProximalSolvers.lambda_max(f, g, β)
    amax = iszero(amax) ? one(T) : amax
    factors = nλ == 1 ? T[amax] : exp.(range(log(amax), log(amax * λmin_ratio); length=nλ))
  end
  a = T.(collect(factors))
  !isempty(a) && all(isfinite, a) && all(>(0), a) && issorted(a; rev=true) ||
    throw(ArgumentError("factors must be positive, finite and nonincreasing"))
  r = ProximalSolvers.solve_path(f, g, a,
    ProximalSolvers.SLOPEAPG(; maxiter=max_iter, tol); x=β, screening)
  bs = [b ./ scales for b in r.solutions]
  b0 = [yb - dot(xb, b) for b in bs]
  all(r.converged) || @warn "some SLOPE path levels did not meet the full optimality tolerance"
  SLOPEPath(a, λ, bs, b0, intercept, length(y), r.converged, r.iterations,
    r.kkt_residuals, r.working_sizes, r.kkt_repairs, info)
end

StatsAPI.coef(s::SLOPESolution) = s.hasintercept ? [s.intercept; s.x] : copy(s.x)
function StatsAPI.predict(s::SLOPESolution, X::AbstractMatrix{<:Real})
  size(X, 2) == length(s.x) || throw(DimensionMismatch("wrong number of predictors"))
  s.intercept .+ X * s.x
end
StatsAPI.residuals(s::SLOPESolution) = s.residuals
StatsAPI.nobs(s::SLOPESolution) = length(s.residuals)
StatsAPI.deviance(s::SLOPESolution) = sum(abs2, s.residuals)
StatsAPI.coef(p::SLOPEPath) = p.intercept ? vcat(transpose(p.β0path), reduce(hcat, p.βpath)) : reduce(hcat, p.βpath)
StatsAPI.nobs(p::SLOPEPath) = p.nobs
function StatsAPI.predict(p::SLOPEPath, X::AbstractMatrix{<:Real})
  size(X, 2) == length(p.lambda) || throw(DimensionMismatch("wrong number of predictors"))
  X * reduce(hcat, p.βpath) .+ transpose(p.β0path)
end
