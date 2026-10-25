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
  xb = intercept ? vec(mean(Z; dims=1)) : zeros(T, p)
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
  if standardize_x
    vals = nonzeros(A)
    for j in 1:p, k in nzrange(A, j)
      vals[k] /= scales[j]
    end
  end
  Z = _SLOPESparseDesign(A, intercept ? xb ./ scales : nothing)
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
end

function _slope_options(max_iter, tol)
  max_iter >= 2 || throw(ArgumentError("max_iter must be at least 2"))
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
end

"""
    slope(X, y, lambda; intercept=false, standardize_x=false, max_iter=10000, tol=1e-7)
    slope(X, y; q=0.1, sigma=1, kwargs...)

Minimize `‖y - β₀ - Xβ‖²/(2n) + ∑ⱼ lambda[j]*|β|₍ⱼ₎` by accelerated
proximal gradient. `lambda` must be nonincreasing and nonnegative. The second
form uses [`slope_bh`](@ref) with `n=size(X,1)`. With `standardize_x`, the
penalty applies to the coefficients of columns scaled to mean square one;
returned coefficients are on the input scale. An intercept centers X and y.
The design is copied to floating-point storage. `SparseMatrixCSC` inputs stay
sparse, including with centering and standardization. Failure of the full
optimality check gives `converged=false` and a warning.
"""
function slope(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::AbstractVector{<:Real};
               intercept::Bool=false, standardize_x::Bool=false, max_iter::Integer=10000, tol::Real=1e-7)
  _slope_options(max_iter, tol)
  Z, v, xb, yb, scales = _slope_data(X, y, intercept, standardize_x)
  β = zeros(eltype(Z), size(Z, 2))
  λ = _slope_lambda(lambda, length(β), eltype(β))
  f = (one(eltype(Z)) / length(v)) * LeastSquaresLoss(v, Z)
  r = ProximalSolvers.solve!(β, f, ProximalBase.SLOPE(λ),
    ProximalSolvers.SLOPEAPG(; maxiter=max_iter, tol))
  it, ok, residual = r.iterations, r.converged, r.kkt_residual
  ok || @warn "SLOPE did not meet its full optimality tolerance" residual iterations=it
  β ./= scales
  β0 = yb - dot(xb, β)
  SLOPESolution(β, β0, intercept, Vector(y .- β0 .- X * β), λ, it, ok, residual)
end

slope(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; q::Real=0.1, sigma::Real=1, kwargs...) =
  slope(X, y, slope_bh(size(X, 2); q, sigma, n=size(X, 1)); kwargs...)

"""
    SLOPEPath(X, y, factors=nothing; lambda=nothing, q=0.1, sigma=1,
              nλ=50, λmin_ratio=0.01, screening=true, intercept=false,
              standardize_x=false, max_iter=10000, tol=1e-7)

Warm-started SLOPE fits with penalties `factor * lambda`. `lambda` defaults
to [`slope_bh`](@ref). Supply positive, nonincreasing `factors`, or a geometric
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
end

function SLOPEPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, factors=nothing;
                   lambda=nothing, q::Real=0.1, sigma::Real=1, nλ::Integer=50, λmin_ratio::Real=0.01,
                   screening::Bool=true, intercept::Bool=false, standardize_x::Bool=false,
                   max_iter::Integer=10000, tol::Real=1e-7)
  _slope_options(max_iter, tol)
  Z, v, xb, yb, scales = _slope_data(X, y, intercept, standardize_x)
  T, p = eltype(Z), size(Z, 2)
  λ = _slope_lambda(lambda === nothing ? slope_bh(p; q, sigma, n=length(y)) : lambda, p, T)
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
    r.kkt_residuals, r.working_sizes, r.kkt_repairs)
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
