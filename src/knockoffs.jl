# Second-order Gaussian and fixed-design equicorrelated knockoffs.
function _knockoff_covariance(Sigma::AbstractMatrix{<:Real})
  p = size(Sigma, 1)
  p > 0 && size(Sigma, 2) == p || throw(DimensionMismatch("Sigma must be nonempty and square"))
  S = Matrix{float(eltype(Sigma))}(Sigma)
  all(isfinite, S) && issymmetric(S) || throw(ArgumentError("Sigma must be finite and symmetric"))
  isposdef(Symmetric(S)) || throw(ArgumentError("Sigma must be positive definite"))
  S
end

"""
    equicorrelated_s(Sigma; safety=1e-8)

Equicorrelated knockoff diagonal: `s[j] = (1-safety) * min(1, 2λmin(R)) *
Sigma[j,j]`, where R is the correlation matrix. This solves the equicorrelated
program `0 ≤ a ≤ 1, 2R - aI ⪰ 0`. The small safety factor, also used by R's
knockoff package, keeps the conditional covariance away from roundoff at the
PSD boundary. Sigma must be positive definite. An SDP solver is not required.
"""
function equicorrelated_s(Sigma::AbstractMatrix{<:Real}; safety::Real=1e-8)
  isfinite(safety) && 0 <= safety < 1 || throw(ArgumentError("safety must be in [0, 1)"))
  S = _knockoff_covariance(Sigma)
  d = sqrt.(diag(S))
  R = S ./ (d * transpose(d))
  a = min(one(eltype(S)), 2eigmin(Symmetric(R))) * (1-safety)
  max(a, zero(a)) .* diag(S)
end

function _knockoff_parameters(S, s)
  d = s === nothing ? equicorrelated_s(S) : Vector{eltype(S)}(s)
  _knockoff_parameters_from_diagonal(S, d)
end

function _knockoff_parameters_from_diagonal(S, d)
  p = size(S, 1)
  length(d) == p || throw(DimensionMismatch("s must have one entry per predictor"))
  all(isfinite, d) && all(>=(0), d) || throw(ArgumentError("s must be finite and nonnegative"))
  # Check the joint covariance on the correlation scale. An absolute tolerance
  # in input units can accept an infeasible diagonal after a change of units.
  scales = sqrt.(diag(S))
  R = S ./ (scales * transpose(scales))
  relative_d = d ./ diag(S)
  all(isfinite, relative_d) || throw(ArgumentError("s is infeasible on the covariance scale"))
  normalized = 2R - Diagonal(relative_d)
  tolerance = 100eps(eltype(S)) * max(2opnorm(R, Inf), norm(relative_d, Inf))
  eigmin(Symmetric(normalized)) >= -tolerance ||
    throw(ArgumentError("s is infeasible: 2Sigma - diag(s) is not PSD"))
  A = cholesky(Symmetric(S)) \ Matrix(Diagonal(d))
  C = Matrix(Symmetric(2Diagonal(d) - Diagonal(d) * A))
  # Reject a materially infeasible diagonal; clip only eigenvalues at roundoff.
  ev = eigen(Symmetric(C))
  minimum(ev.values) >= -100eps(eltype(S)) * max(opnorm(2Diagonal(d), Inf), opnorm(Diagonal(d) * A, Inf)) ||
    throw(ArgumentError("s is infeasible: 2diag(s) - diag(s)*inv(Sigma)*diag(s) is not PSD"))
  F = cholesky(Symmetric(C); check=false)
  root = issuccess(F) ? Matrix(F.U) : Diagonal(sqrt.(max.(ev.values, 0))) * transpose(ev.vectors)
  d, A, root
end

"""
    gaussian_knockoffs(X, mu, Sigma; rng=Random.default_rng(), s=nothing, noise=nothing)

Gaussian model-X knockoffs `X - (X-mu')*(Sigma\\diag(s)) + Z*C`, with
`C'C = 2diag(s) - diag(s)*(Sigma\\diag(s))`. By default s is
[`equicorrelated_s`](@ref). Rows of X must be independent Gaussian draws with
population mean mu and covariance Sigma for exact model-X FDR control.
Plug-in estimates give only second-order matching, without that guarantee.
`noise` optionally supplies an n-by-p matrix of independent standard normals
for reproducible comparisons; otherwise Z is drawn from rng. It must be
independent of X and y. Neither X nor the supplied parameters are modified.
`strategy`/`construction_options` select the diagonal through
[`knockoff_diagonal`](@ref); the returned object remains a matrix.
"""
function gaussian_knockoffs(X::AbstractMatrix{<:Real}, mu::AbstractVector{<:Real},
                            Sigma::AbstractMatrix{<:Real}; rng::AbstractRNG=default_rng(), s=nothing, noise=nothing,
                            strategy=nothing, construction_options=(;))
  first(_gaussian_knockoffs_details(X, mu, Sigma; rng, s, noise, strategy, construction_options))
end

function _gaussian_knockoffs_details(X::AbstractMatrix{<:Real}, mu::AbstractVector{<:Real},
                            Sigma::AbstractMatrix{<:Real}; rng::AbstractRNG=default_rng(), s=nothing, noise=nothing,
                            strategy=nothing, construction_options=(;))
  _knockoff_strategy(strategy, s, construction_options)
  S = _knockoff_covariance(Sigma)
  n, p = size(X)
  n > 0 && p == size(S, 1) && length(mu) == p || throw(DimensionMismatch("X, mu and Sigma have incompatible dimensions"))
  all(isfinite, X) && all(isfinite, mu) || throw(ArgumentError("X and mu must be finite"))
  plan = _knockoff_choose_diagonal(S, strategy, s, construction_options)
  _, A, C = _knockoff_parameters_from_diagonal(S, plan.s)
  T = float(promote_type(eltype(X), eltype(S), eltype(mu)))
  Z = noise === nothing ? randn(rng, T, n, p) : Matrix{T}(noise)
  size(Z) == (n, p) || throw(DimensionMismatch("noise must have the dimensions of X"))
  all(isfinite, Z) || throw(ArgumentError("noise must be finite"))
  X - (X .- transpose(mu)) * A + Z * C, _knockoff_construction_info(plan)
end

"""
    fixed_knockoffs(X; s=nothing)

Fixed-X knockoffs for a full-column-rank design with n ≥ 2p. Uses the
orthogonal complement from QR and the equicorrelated diagonal by default.
The output satisfies `Xk'Xk = X'X`, `X'Xk = X'X - diag(s)`, to roundoff.
Columns need not have unit norm; s is computed on the correlation scale and
mapped back to the Gram matrix. For fixed-X FDR control, use a linear model without an intercept, independent
homoscedastic Gaussian errors, and a swap-antisymmetric statistic with the
sufficiency property, such as a lasso fit at a prespecified penalty level.
`strategy`/`construction_options` select the diagonal through
[`knockoff_diagonal`](@ref); the returned object remains a matrix.
"""
function fixed_knockoffs(X::AbstractMatrix{<:Real}; s=nothing, strategy=nothing, construction_options=(;))
  first(_fixed_knockoffs_details(X; s, strategy, construction_options))
end

function _fixed_knockoffs_details(X::AbstractMatrix{<:Real}; s=nothing, strategy=nothing, construction_options=(;))
  _knockoff_strategy(strategy, s, construction_options)
  n, p = size(X)
  p > 0 && n >= 2p || throw(ArgumentError("fixed-X knockoffs require n ≥ 2p > 0"))
  Z = Matrix{float(eltype(X))}(X)
  all(isfinite, Z) || throw(ArgumentError("X must be finite"))
  S = _knockoff_covariance(transpose(Z) * Z)
  plan = _knockoff_choose_diagonal(S, strategy, s, construction_options)
  _, A, C = _knockoff_parameters_from_diagonal(S, plan.s)
  F = qr(Z)
  E = zeros(eltype(Z), n, p)
  for j in 1:p
    E[p+j, j] = 1
  end
  U = F.Q * E
  Z - Z * A + U * C, _knockoff_construction_info(plan)
end

"""
    knockoff_threshold(W; q=0.1, offset=1)

Smallest positive observed |W| with `(offset + count(W ≤ -t)) /
max(1, count(W ≥ t)) ≤ q`, or Inf if none exists. `offset=1` is knockoff+,
which controls FDR under the construction/statistic assumptions; `offset=0`
is the original knockoff rule and controls a modified FDR. Ties are inclusive.
"""
function knockoff_threshold(W::AbstractVector{<:Real}; q::Real=0.1, offset::Integer=1)
  isfinite(q) && 0 < q < 1 || throw(ArgumentError("q must be in (0, 1)"))
  offset in (0, 1) || throw(ArgumentError("offset must be zero or one"))
  all(isfinite, W) || throw(ArgumentError("statistics must be finite"))
  order = sortperm(abs.(W))
  pos, neg = count(>(0), W), count(<(0), W)
  k = 1
  while k <= length(W)
    t = abs(W[order[k]])
    if t > 0 && (offset + neg) / max(1, pos) <= q
      return float(t)
    end
    while k <= length(W) && abs(W[order[k]]) == t
      v = W[order[k]]
      pos -= v > 0; neg -= v < 0
      k += 1
    end
  end
  Inf
end

"""
    lasso_coefdiff(X, Xk, y, lambda; options=CDOptions(), kwargs...)

Coefficient-difference statistic `abs(β[1:p]) - abs(β[p+1:2p])` from a
joint lasso on `[X Xk]`, at a prespecified finite positive lambda. Fits without
an intercept or column standardization to preserve fixed-X sufficiency. Scale X
before construction if needed. To remove an intercept, project X and y onto an
orthonormal basis orthogonal to the constant vector before construction;
subtracting means alone is insufficient. Additional
keywords are CDOptions fields. Failure to converge raises an error.
"""
function lasso_coefdiff(X::AbstractMatrix{<:Real}, Xk::AbstractMatrix{<:Real},
                        y::AbstractVector{<:Real}, lambda::Real; options::CDOptions=CDOptions(), rng::AbstractRNG=default_rng(), kwargs...)
  size(X) == size(Xk) || throw(DimensionMismatch("X and Xk must have the same dimensions"))
  isfinite(lambda) && lambda > 0 || throw(ArgumentError("lambda must be finite and positive"))
  opts = _options(options, kwargs)
  sol = lasso(hcat(X, Xk), y, lambda, opts; rng)
  sol.converged || error("joint knockoff lasso failed to converge")
  b = Vector(sol.x)
  p = size(X, 2)
  abs.(b[1:p]) - abs.(b[p+1:2p])
end

"""
    KnockoffResult

Result of [`knockoff_filter`](@ref), with `selected` (column indices), `W`,
`threshold`, `Xk`, target `q`, `lambda`, model `method`, and `construction`
metadata with the actual diagonal strategy and optimizer diagnostics.
"""
struct KnockoffResult{T, M<:AbstractMatrix}
  selected::Vector{Int}
  W::Vector{T}
  threshold::T
  Xk::M
  q::T
  lambda::T
  method::Symbol
  construction::NamedTuple
end

KnockoffResult(selected, W, threshold, Xk, q, lambda, method) =
  KnockoffResult(selected, W, threshold, Xk, q, lambda, method,
    (strategy=:unspecified, s=nothing, diagnostics=(backend=:unspecified,)))

KnockoffResult{T,M}(selected, W, threshold, Xk, q, lambda, method) where {T,M<:AbstractMatrix} =
  KnockoffResult{T,M}(selected, W, threshold, Xk, q, lambda, method,
    (strategy=:unspecified, s=nothing, diagnostics=(backend=:unspecified,)))

"""
    knockoff_filter(X, y, lambda; method=:gaussian, mu=nothing, Sigma=nothing,
                    Xk=nothing, q=0.1, rng=Random.default_rng(), options=CDOptions(), kwargs...)

Construct knockoffs, fit [`lasso_coefdiff`](@ref), and select with the
knockoff+ threshold. `method=:gaussian` requires population mu and Sigma;
`:fixed` uses [`fixed_knockoffs`](@ref). Supplying Xk requires `method=:provided`
and makes the caller responsible for the construction assumptions. lambda
must be chosen independently of y for the stated fixed-X guarantee. `kwargs`
are CDOptions fields, not model preprocessing options. Exact FDR control also
requires swap antisymmetry; ties/nonunique lasso solutions need care.
"""
function knockoff_filter(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::Real;
                         method::Symbol=:gaussian, mu=nothing, Sigma=nothing, Xk=nothing,
                         q::Real=0.1, rng::AbstractRNG=default_rng(), options::CDOptions=CDOptions(),
                         strategy=nothing, s=nothing, noise=nothing, construction_options=(;), kwargs...)
  # Validate q before doing a potentially expensive fit.
  knockoff_threshold(Float64[]; q)
  method in (:gaussian, :fixed, :provided) || throw(ArgumentError("method must be :gaussian, :fixed or :provided"))
  Xk === nothing || method === :provided || throw(ArgumentError("use method=:provided when supplying Xk"))
  construction_options isa NamedTuple || throw(ArgumentError("construction_options must be a NamedTuple"))
  method === :gaussian || noise === nothing || throw(ArgumentError("noise is only used with method=:gaussian"))
  if method === :provided
    strategy === nothing && s === nothing && isempty(construction_options) ||
      throw(ArgumentError("provided Xk conflicts with strategy, s or construction_options"))
  else
    _knockoff_strategy(strategy, s, construction_options)
  end
  isfinite(lambda) && lambda > 0 || throw(ArgumentError("lambda must be finite and positive"))
  # Validate solver options before optimization/sampling and keep construction
  # options out of CDOptions. Existing supplied-noise/RNG semantics are native.
  opts = _options(options, kwargs)
  K, construction = if method === :gaussian
    mu !== nothing && Sigma !== nothing || throw(ArgumentError("Gaussian knockoffs require mu and Sigma"))
    _gaussian_knockoffs_details(X, mu, Sigma; rng, s, noise, strategy, construction_options)
  elseif method === :fixed
    _fixed_knockoffs_details(X; s, strategy, construction_options)
  else
    Xk === nothing && throw(ArgumentError("method=:provided requires Xk"))
    (Xk, (strategy=:provided, s=nothing, diagnostics=(backend=:provided, feasibility=:caller_responsibility)))
  end
  # Only solver options are accepted here; model preprocessing can destroy
  # sufficiency of a fixed-design statistic after knockoffs are constructed.
  W = lasso_coefdiff(X, K, y, lambda; options=opts, rng)
  t = convert(eltype(W), knockoff_threshold(W; q))
  selected = isfinite(t) ? findall(>=(t), W) : Int[]
  KnockoffResult(selected, W, t, K, eltype(W)(q), eltype(W)(lambda), method, construction)
end
