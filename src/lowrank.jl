######################################################################
# Low-rank multivariate regression
######################################################################

"""
    LowRankSolution

A multivariate regression fit. `λ` is its penalty level, `B` the coefficient
matrix, `intercept` the response intercepts, `rank` the numerical rank of `B`, and `activerows` the
selected predictors. `iterations`, `converged` and `objective` describe the
optimization (`:sparse_svd` reports the combined residual loss); a nonconvex
fit reports convergence to a block stationary point,
not a global minimum. `coef`, `predict`, `residuals`, `nobs`, `deviance`, `dof`,
`aic` and `bic` apply. `dof` counts the parameters of the selected rank/support
model; it is not a Stein unbiased degrees-of-freedom estimate.
"""
struct LowRankSolution{T<:AbstractFloat}
  B::Matrix{T}
  intercept::Vector{T}
  rank::Int
  activerows::Vector{Int}
  iterations::Int
  converged::Bool
  objective::T
  residuals::Matrix{T}
  hasintercept::Bool
  df::Int
  method::Symbol
  df_exact::T
  λ::T
end

function _lr_data(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, intercept::Bool)
  n, p = size(X)
  n > 0 && p > 0 && size(Y, 2) > 0 || throw(ArgumentError("X and Y must have nonempty dimensions"))
  size(Y, 1) == n || throw(DimensionMismatch("X and Y must have the same number of rows"))
  T = float(promote_type(eltype(X), eltype(Y)))
  T <: Union{Float32, Float64} || throw(ArgumentError("low-rank regression supports Float32 and Float64 data"))
  A, Z = Matrix{T}(X), Matrix{T}(Y)
  _check_finite(A, "X"); _check_finite(Z, "Y")
  mx = intercept ? vec(mean(A; dims = 1)) : zeros(T, p)
  my = intercept ? vec(mean(Z; dims = 1)) : zeros(T, size(Z, 2))
  A .-= mx'; Z .-= my'
  (; X = A, Y = Z, mx, my, intercept)
end

function _lr_svd(d; rtol = nothing)
  F = svd(d.X; full = false)
  T = eltype(d.X)
  τ = rtol === nothing ? max(size(d.X)...) * eps(T) : T(rtol)
  isfinite(τ) && τ >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
  k = count(>(τ * first(F.S)), F.S)
  U, V, s = F.U[:, 1:k], F.V[:, 1:k], F.S[1:k]
  H = svd(U' * d.Y; full = false)
  # rank(X) uses its default tolerance, independently of the user's fitting
  # cutoff. Reuse these singular values for fixed-rank result diagnostics.
  diagnostic_xrank = count(>(min(size(d.X)...) * eps(T) * first(F.S)), F.S)
  (; U, V, s, H, xrank = k, diagnostic_xrank)
end

# Stein degrees of freedom of rank-constrained fitted values. A truncation
# through tied singular values is not differentiable and has no finite score.
function _lr_exact_df(f, r, q, intercept)
  base = r * (f.xrank + q - r) + q * intercept
  correction = zero(eltype(f.s))
  for i in 1:r, j in r+1:length(f.H.S)
    # Normalize before squaring: the Stein score is invariant to response
    # scale, and raw squares can underflow/overflow on otherwise valid data.
    si = f.H.S[i]
    iszero(si) && return Inf
    ratio = f.H.S[j] / si
    gap = one(ratio) - ratio^2
    gap > eps(eltype(f.s)) || return Inf
    correction += 2ratio^2 / gap
  end
  base + correction
end

function _lr_rank(r, maximum)
  r isa Integer && 0 <= r <= maximum || throw(ArgumentError("rank must be an integer in 0:$maximum"))
  Int(r)
end

function _lr_rrcoef(d, f, r)
  r == 0 && return zeros(eltype(d.X), size(d.X, 2), size(d.Y, 2))
  L, R = _lr_rrfactors(f, r)
  L * R
end
_lr_rrfactors(f, r) = (f.V * (f.H.U[:, 1:r] ./ f.s), f.H.S[1:r] .* f.H.Vt[1:r, :])

function _lr_solution(d, B, λ, iterations, converged, objective, method;
                      df = nothing, df_exact = NaN, rows = _activerows(B),
                      coefficient_rank = nothing, residual = nothing)
  r = coefficient_rank === nothing ? LinearAlgebra.rank(B) : coefficient_rank
  q = size(B, 2)
  if df === nothing
    k = isempty(rows) ? 0 : LinearAlgebra.rank(d.X[:, rows])
    df = r * (k + q - r) + q * d.intercept
  end
  E = residual === nothing ? d.Y - d.X * B : residual
  LowRankSolution(B, d.my - B' * d.mx, r, rows, iterations, converged,
                  eltype(B)(objective), E, d.intercept, df, method, eltype(B)(df_exact), eltype(B)(λ))
end

function _lr_rrsolution(d, f, bound)
  T = eltype(d.X)
  p, q = size(d.X, 2), size(d.Y, 2)
  if bound == 0
    B, r = zeros(T, p, q), 0
  else
    L, R = _lr_rrfactors(f, bound)
    B = L * R
    # The singular values of B equal those of this bound-by-bound core.
    # Apply rank(B)'s original dimension-dependent cutoff, including when
    # the fitted coefficient rank is smaller than the requested bound.
    core = Matrix(qr(L).R) * Matrix(qr(R').R)'
    σ = svdvals!(core)
    cutoff = min(p, q) * eps(T) * first(σ)
    # Near the numerical cutoff, roundoff in L*R itself can affect rank.
    # Retain the original dense-matrix diagnostic for these ambiguous cases.
    ambiguous = any(<=(8cutoff), σ) || 8bound*eps(T)*norm(L)*norm(R) > cutoff
    r = ambiguous ? LinearAlgebra.rank(B) : count(>(cutoff), σ)
  end
  rows = _activerows(B)
  k = isempty(rows) ? 0 : length(rows) == p ? f.diagnostic_xrank : LinearAlgebra.rank(d.X[:, rows])
  df = r * (k + q - r) + q * d.intercept
  E = d.Y - d.X * B
  _lr_solution(d, B, 0, 0, true, sum(abs2, E)/(2size(d.X, 1)), :rank;
    df, df_exact = _lr_exact_df(f, bound, q, d.intercept), rows, coefficient_rank = r, residual = E)
end

"""
    reduced_rank(X, Y, rank; intercept = true, rtol = nothing)

Minimize `‖Y - 1β₀' - XB‖²_F` subject to `rank(B) ≤ rank`. The truncated
SVD of the least-squares fitted values gives the solution (Anderson, 1951;
Izenman, 1975). A compact SVD of `X` handles wide and rank-deficient designs
and chooses the minimum-norm coefficient matrix. `rtol` is the relative
singular-value cutoff for `X`, by default `max(size(X)...) * eps(T)`.
The intercept is unpenalized; columns are kept on their original scales.
"""
function reduced_rank(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, r::Integer;
                      intercept::Bool = true, rtol = nothing)
  d = _lr_data(X, Y, intercept)
  f = _lr_svd(d; rtol)
  _lr_rank(r, min(f.xrank, size(Y, 2)))
  _lr_rrsolution(d, f, r)
end

"""
    select_rank(X, Y; sigma2 = nothing, multiplier = 2, μ = nothing, intercept = true)

The rank selection criterion of Bunea, She and Wegkamp (2011): minimize
`RSS + μ rank(B)`. Keep fitted singular values whose squares exceed `μ`.
By default `μ = multiplier * sigma2 * (√rank(Xc) + √q)^2`, with centered
`Xc` when fitting an intercept. If omitted, `sigma2` is estimated from the
full least-squares residuals using `q * (n - rank(Xc) - intercept)` degrees
of freedom. A saturated design requires a supplied variance or `μ`.
Returns `(solution, rank, μ, sigma2, singularvalues)`.
"""
function select_rank(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                     sigma2 = nothing, multiplier::Real = 2, μ = nothing,
                     intercept::Bool = true, rtol = nothing)
  d = _lr_data(X, Y, intercept)
  f = _lr_svd(d; rtol)
  isfinite(multiplier) && multiplier > 0 || throw(ArgumentError("multiplier must be finite and positive"))
  if μ === nothing
    if sigma2 === nothing
      denom = size(Y, 2) * (size(X, 1) - f.xrank - intercept)
      denom > 0 || throw(ArgumentError("saturated design: supply sigma2 or μ"))
      sigma2 = sum(abs2, d.Y - f.U * (f.U' * d.Y)) / denom
    end
    isfinite(sigma2) && sigma2 >= 0 || throw(ArgumentError("sigma2 must be finite and nonnegative"))
    μ = multiplier * sigma2 * (sqrt(f.xrank) + sqrt(size(Y, 2)))^2
  else
    isfinite(μ) && μ >= 0 || throw(ArgumentError("μ must be finite and nonnegative"))
  end
  r = count(s -> s^2 > μ, f.H.S)
  sol = _lr_rrsolution(d, f, r)
  (; solution = sol, rank = r, μ, sigma2, singularvalues = copy(f.H.S))
end

function _lr_options(λ, max_iter, tol)
  isfinite(λ) && λ >= 0 || throw(ArgumentError("λ must be finite and nonnegative"))
  max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
end

# A shared gradient setup for every level of a nuclear-norm path. Use the
# data form for a wide design, avoiding a p × p allocation there.
function _lr_gradient(d)
  n, p = size(d.X)
  L = opnorm(d.X)^2 / n
  A = p <= n ? d.X' * d.X / n : nothing
  C = d.X' * d.Y / n
  (; A, C, L)
end
_lr_grad(d, g, B) = g.A === nothing ? d.X' * (d.X * B) / size(d.X, 1) - g.C : g.A * B - g.C

function _lr_nuclear(d, g, λ, start; max_iter, tol, solver = :apg)
  if solver == :panoc
    ext = Base.get_extension(@__MODULE__, :HighDimensionalModelsProximalAlgorithmsExt)
    ext === nothing && throw(ArgumentError("solver = :panoc requires ProximalAlgorithms; install and load it with `using ProximalAlgorithms`"))
    return ext.nuclear_panoc(d, g, λ, start; max_iter, tol)
  end
  T = eltype(d.X)
  B, Z = copy(start), copy(start)
  g.L == 0 && return _lr_solution(d, zero(B), λ, 0, true, sum(abs2, d.Y) / (2size(d.X, 1)), :nuclear)
  γ = inv(g.L)
  penalty = NuclearNorm(T(λ))
  candidate, step = similar(B), similar(B)
  t = one(T)
  scale = max(one(T), norm(g.C))
  converged, iteration = false, 0
  for k in 1:max_iter
    iteration = k
    G = _lr_grad(d, g, Z)
    @. step = Z - γ * G
    prox!(candidate, penalty, step, γ)
    # Certify the gradient mapping at the point actually returned. The
    # extrapolated-point mapping is insufficient on ill-conditioned designs.
    if norm(candidate - B) <= sqrt(T(tol)) * max(one(T), norm(B)) || k == max_iter
      H = _lr_grad(d, g, candidate)
      @. step = candidate - γ * H
      prox!(step, penalty, step, γ)
      if norm(step - candidate) / γ <= tol * scale
        B .= candidate
        converged = true
        break
      end
    end
    tn = (one(T) + sqrt(one(T) + 4t^2)) / 2
    # Adaptive restart of the accelerated proximal gradient iteration.
    β = (t - 1) / tn
    if dot(Z - candidate, candidate - B) > 0
      tn = one(T)
      β = zero(T)
    end
    @. Z = candidate + β * (candidate - B)
    B .= candidate
    t = tn
  end
  objective = sum(abs2, d.Y - d.X * B) / (2size(d.X, 1)) + penalty(B)
  _lr_solution(d, B, λ, iteration, converged, objective, :nuclear)
end

"""
    nuclear_regression(X, Y, λ; solver = :apg, intercept = true, max_iter = 10_000, tol = 1e-7)

Minimize `‖Y - 1β₀' - XB‖²_F / (2n) + λ ‖B‖_*` by accelerated proximal
gradient, using `ProximalBase.NuclearNorm` for singular-value thresholding.
Columns remain on their original scales. `tol` bounds the Frobenius norm of
the proximal gradient mapping relative to `max(1, ‖Xc'Yc/n‖_F)`.
See [`LowRankPath`](@ref) for warm starts over several penalties.
`solver = :panoc` uses ProximalAlgorithms' PANOC with the same loss, nuclear
proximal operator, and stopping certificate; first load `ProximalAlgorithms`.
"""
function nuclear_regression(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, λ::Real;
                            solver::Symbol = :apg, intercept::Bool = true,
                            max_iter::Integer = 10_000, tol::Real = 1e-7)
  solver in (:apg, :panoc) || throw(ArgumentError("solver must be :apg or :panoc"))
  _lr_options(λ, max_iter, tol)
  d = _lr_data(X, Y, intercept)
  _lr_nuclear(d, _lr_gradient(d), λ, zeros(eltype(d.X), size(X, 2), size(Y, 2)); max_iter, tol, solver)
end

function _lr_group_kkt(A, G, λ, w)
  err = zero(eltype(A))
  for j in axes(A, 1)
    a = view(A, j, :)
    na = norm(a)
    err = max(err, na == 0 ? max(0, norm(view(G, j, :)) - λ*w[j]) : norm(view(G, j, :) + (λ*w[j]/na).*a))
  end
  err
end

# Exact row coordinate descent for the convex A block of B = A V', V'V = I.
# The inner stopping rule checks its KKT equations, including the zero rows.
function _lr_group!(A, XX, C, λ, w; max_iter, tol)
  G = XX * A - C
  scale = max(one(eltype(A)), norm(C))
  for sweep in 1:max_iter
    for j in axes(A, 1)
      h = XX[j, j]
      old = copy(A[j, :])
      z = h .* old .- G[j, :]
      nz = norm(z)
      A[j, :] .= h > 0 && nz > λ * w[j] ? ((1 - λ * w[j] / nz) / h) .* z : zero(old)
      change = A[j, :] - old
      G .+= view(XX, :, j) * change'
    end
    err = _lr_group_kkt(A, G, λ, w)
    err <= tol * scale && return true
  end
  false
end

function _lr_group_fista!(A, XX, C, λ, w; max_iter, tol, L)
  T = eltype(A)
  scale = max(one(T), norm(C))
  if iszero(L)
    fill!(A, zero(T))
    return _lr_group_kkt(A, -C, λ, w) <= tol * scale
  end
  Z, W, G = copy(A), similar(A), similar(A)
  t = one(T)
  for k in 1:max_iter
    mul!(G, XX, Z)
    @. W = Z - (G - C) / L
    for j in axes(W, 1)
      row = view(W, j, :)
      nr = norm(row)
      row .*= nr == 0 ? zero(T) : max(zero(T), 1 - λ*w[j]/(L*nr))
    end
    if k % 10 == 0 || k == max_iter
      mul!(G, XX, W)
      G .-= C
      if _lr_group_kkt(W, G, λ, w) <= tol * scale
        A .= W
        return true
      end
    end
    restart = zero(T)
    @inbounds for j in eachindex(A)
      restart += (Z[j] - W[j]) * (W[j] - A[j])
    end
    tn = restart > 0 ? one(T) : (one(T) + sqrt(one(T) + 4t^2))/2
    β = restart > 0 ? zero(T) : (t - 1)/tn
    @. Z = W + β * (W - A)
    A .= W
    t = tn
  end
  false
end

function _lr_sparse(d, r, λ, w, initial; max_iter, tol, inner_max_iter, inner_tol, block_solver = :cd)
  T = eltype(d.X)
  n, p = size(d.X)
  r == 0 && return _lr_solution(d, zeros(T, p, size(d.Y, 2)), λ, 0, true, sum(abs2, d.Y)/(2n), :sparse)
  F = svd(initial; full = false)
  V = F.V[:, 1:r]
  A = initial * V
  XX, XY = d.X' * d.X / n, d.X' * d.Y / n
  L = block_solver == :fista ? eigmax(Symmetric(XX)) : zero(T)
  B = A * V'
  iteration, converged = 0, false
  for k in 1:max_iter
    iteration = k
    old = B
    C = XY * V
    inner = block_solver == :fista ?
      _lr_group_fista!(A, XX, C, λ, w; max_iter = inner_max_iter, tol = inner_tol, L) :
      _lr_group!(A, XX, C, λ, w; max_iter = inner_max_iter, tol = inner_tol)
    S = svd(XY' * A; full = false)
    V = S.U * S.Vt
    B = A * V'
    if inner && norm(B - old) <= tol * max(one(T), norm(B))
      # The response subspace just changed: certify the A block at this V,
      # not only at the previous subspace used by coordinate descent.
      C = XY * V
      if _lr_group_kkt(A, XX * A - C, λ, w) <= inner_tol * max(one(T), norm(C))
        converged = true
        break
      end
    end
  end
  objective = sum(abs2, d.Y - d.X * B) / (2n) + λ * sum(w[j] * norm(view(A, j, :)) for j in 1:p)
  _lr_solution(d, B, λ, iteration, converged, objective, :sparse)
end

function _lr_weights(d, initial, weights, adaptive, gamma)
  isfinite(gamma) && gamma >= 0 || throw(ArgumentError("gamma must be finite and nonnegative"))
  T = eltype(d.X)
  p = size(d.X, 2)
  w = weights === nothing ? (adaptive ? [inv(max(norm(view(initial, j, :)), sqrt(eps(T)))^gamma) for j in 1:p] : ones(T, p)) : T.(weights)
  length(w) == p || throw(DimensionMismatch("weights must have one entry per predictor"))
  all(x -> isfinite(x) && x > 0, w) || throw(ArgumentError("weights must be finite and positive"))
  w
end

"""
    sparse_reduced_rank(X, Y, rank, λ; intercept = true, weights = nothing,
                        adaptive = false, gamma = 2, max_iter = 1000, tol = 1e-7,
                        inner_max_iter = 10_000, inner_tol = 1e-9, block_solver = :cd)

The row-sparse reduced-rank regression of Chen and Huang (2012): minimize
`‖Y - 1β₀' - XB‖²_F/(2n) + λ ∑ⱼ weights[j] ‖B[j,:]‖₂`, with rank at most
`rank`. Alternates a group-lasso fit of `A` and an orthogonal Procrustes update
of `V` in `B = AV'`. It starts from reduced-rank least squares. This problem
is nonconvex; convergence is local. `adaptive = true` uses inverse row norms
of the initial RRR fit to power `gamma`, with a `sqrt(eps(T))` floor.
Explicit positive `weights` override adaptive weights. No column rescaling is
performed. For rrpack's unscaled loss use `λ_rrpack = n * λ`.
`block_solver = :fista` replaces row coordinate descent with restarted block
FISTA, using the same weighted group proximal map and KKT stopping rule.
"""
function sparse_reduced_rank(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, r::Integer, λ::Real;
                             intercept::Bool = true, weights = nothing, adaptive::Bool = false,
                             gamma::Real = 2, max_iter::Integer = 1000, tol::Real = 1e-7,
                             inner_max_iter::Integer = 10_000, inner_tol::Real = 1e-9,
                             block_solver::Symbol = :cd)
  block_solver in (:cd, :fista) || throw(ArgumentError("block_solver must be :cd or :fista"))
  _lr_options(λ, max_iter, tol); _lr_options(λ, inner_max_iter, inner_tol)
  d = _lr_data(X, Y, intercept)
  f = _lr_svd(d)
  _lr_rank(r, min(f.xrank, size(Y, 2)))
  initial = _lr_rrcoef(d, f, r)
  w = _lr_weights(d, initial, weights, adaptive, gamma)
  _lr_sparse(d, r, λ, w, initial; max_iter, tol, inner_max_iter, inner_tol, block_solver)
end

"""
    lowrank_ic(solution; criterion = :bic, df = :naive)
    lowrank_ic(path; criterion = :bic)

Gaussian model-selection scores `log(RSS) + penalty * dof/(n*q)`;
`:aic` uses 2, `:bic` uses `log(n*q)`, and `:bicp` uses
`2log(n*q)` for row-sparse regression and `2log(p*q)` otherwise, as in rrpack. Degrees of freedom count the selected
rank/support model. Smaller is better. Penalized fits use a parameter-count
approximation rather than exact Stein degrees of freedom. `df = :exact`
uses the Stein formula for rank-constrained fits (Mukherjee et al., 2015);
a cutoff through tied singular values has infinite score.
"""
function lowrank_ic(s::LowRankSolution; criterion::Symbol = :bic, df::Symbol = :naive)
  n, q = size(s.residuals)
  p = size(s.B, 1)
  penalty = criterion == :bic ? log(n*q) : criterion == :aic ? 2 : criterion == :bicp ? 2log(s.method == :sparse ? n*q : p*q) : throw(ArgumentError("criterion must be :aic, :bic or :bicp"))
  df in (:naive, :exact) || throw(ArgumentError("df must be :naive or :exact"))
  df == :exact && s.method != :rank && throw(ArgumentError("exact degrees of freedom are available only for rank-constrained fits"))
  degrees = df == :exact ? s.df_exact : s.df
  log(sum(abs2, s.residuals)) + penalty * degrees / (n*q)
end

"""
    LowRankPath(X, Y; method = :nuclear, λpath = nothing, ranks = nothing,
                rank = 1, nλ = 50, λminratio = 0.01, kwargs...)

Fit a path with `method = :nuclear`, `:sparse`, `:rank`, or `:sparse_svd`. Nuclear and sparse
paths use decreasing `λpath` (or an automatic log grid); `:sparse` holds `rank`
fixed. `:rank` fits the supplied `ranks`, by default zero through the maximum
estimable rank. `:sparse_svd` fits the supplied rank bounds with component
penalties chosen by BIC. `solutions` contains independent [`LowRankSolution`](@ref)s.
Nuclear paths reuse the gradient setup and warm start the coefficient matrix.
`solver = :panoc` selects the optional ProximalAlgorithms backend for nuclear
paths. `block_solver = :fista` selects block FISTA for sparse paths; defaults
are `solver = :apg` and `block_solver = :cd`.
Sparse paths warm start nonzero fits, restarting at the RRR initialization
after a zero fit to avoid an absorbing zero block. Adaptive weights are fixed
from the training-data RRR fit over the whole path. Select using
`argmin(lowrank_ic(path))`, or [`CVLowRankPath`](@ref).
"""
struct LowRankPath{T<:AbstractFloat}
  solutions::Vector{LowRankSolution{T}}
  λpath::Vector{T}
  ranks::Vector{Int}
  method::Symbol
end

function LowRankPath(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                     method::Symbol = :nuclear, λpath = nothing, ranks = nothing, rank::Integer = 1,
                     nλ::Integer = 50, λminratio::Real = 0.01, intercept::Bool = true,
                     max_iter::Integer = 10_000, tol::Real = 1e-7, weights = nothing,
                     adaptive::Union{Nothing, Bool} = nothing, gamma::Real = 2, criterion::Symbol = :bic,
                     inner_max_iter::Integer = 10_000, inner_tol::Real = 1e-9,
                     solver::Symbol = :apg, block_solver::Symbol = :cd)
  method in (:nuclear, :sparse, :rank, :sparse_svd) || throw(ArgumentError("method must be :nuclear, :sparse, :rank or :sparse_svd"))
  solver in (:apg, :panoc) || throw(ArgumentError("solver must be :apg or :panoc"))
  block_solver in (:cd, :fista) || throw(ArgumentError("block_solver must be :cd or :fista"))
  method == :nuclear || solver == :apg || throw(ArgumentError("solver applies only to method = :nuclear"))
  method == :sparse || block_solver == :cd || throw(ArgumentError("block_solver applies only to method = :sparse"))
  adaptive = adaptive === nothing ? method == :sparse_svd : adaptive
  _lr_options(0, max_iter, tol); _lr_options(0, inner_max_iter, inner_tol)
  d = _lr_data(X, Y, intercept)
  T = eltype(d.X)
  sols = LowRankSolution{T}[]
  if method in (:rank, :sparse_svd)
    λpath === nothing || throw(ArgumentError("a rank path takes ranks, not λpath"))
    f = _lr_svd(d)
    rs = ranks === nothing ? collect(0:min(f.xrank, size(Y, 2))) : collect(ranks)
    isempty(rs) && throw(ArgumentError("ranks must be nonempty"))
    for r in rs
      _lr_rank(r, min(f.xrank, size(Y, 2)))
      if method == :sparse_svd
        push!(sols, sparse_svd_regression(X, Y, r; intercept, nλ, λminratio,
          adaptive, gamma, criterion, max_iter, tol, inner_max_iter, inner_tol).solution)
      else
        push!(sols, _lr_rrsolution(d, f, r))
      end
    end
    return LowRankPath(sols, zeros(T, length(rs)), Int.(rs), method)
  end
  ranks === nothing || throw(ArgumentError("ranks is only used for method = :rank"))
  initial = zeros(T, size(X, 2), size(Y, 2))
  w = ones(T, size(X, 2))
  if method == :sparse
    f = _lr_svd(d)
    _lr_rank(rank, min(f.xrank, size(Y, 2)))
    initial = _lr_rrcoef(d, f, rank)
    w = _lr_weights(d, initial, weights, adaptive, gamma)
  elseif weights !== nothing || adaptive
    throw(ArgumentError("row weights apply only to method = :sparse"))
  end
  g = method == :nuclear ? _lr_gradient(d) : nothing
  if λpath === nothing
    nλ >= 1 || throw(ArgumentError("nλ must be positive"))
    isfinite(λminratio) && 0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
    C = d.X' * d.Y / size(X, 1)
    λmax = method == :nuclear ? opnorm(C) : maximum(norm(view(C, j, :))/w[j] for j in axes(C, 1))
    λmax *= 1 + sqrt(eps(T))
    λs = T[λmax * λminratio^(k/max(1, nλ-1)) for k in 0:nλ-1]
  else
    λs = T.(collect(λpath))
    isempty(λs) && throw(ArgumentError("λpath must be nonempty"))
    all(x -> isfinite(x) && x >= 0, λs) || throw(ArgumentError("λpath must be finite and nonnegative"))
    issorted(λs; rev = true) || throw(ArgumentError("λpath must be decreasing"))
  end
  start = copy(initial)
  for λ in λs
    s = method == :nuclear ? _lr_nuclear(d, g, λ, start; max_iter, tol, solver) :
      _lr_sparse(d, rank, λ, w, start; max_iter, tol, inner_max_iter, inner_tol, block_solver)
    push!(sols, s)
    start = s.rank == 0 ? copy(initial) : copy(s.B)
  end
  LowRankPath(sols, λs, [s.rank for s in sols], method)
end
lowrank_ic(p::LowRankPath; kwargs...) = [lowrank_ic(s; kwargs...) for s in p.solutions]

"""
    CVLowRankPath(X, Y; nfolds = 5, folds = nothing, rng = Random.default_rng(), kwargs...)

K-fold cross-validation of [`LowRankPath`](@ref), on a common penalty or rank
grid. Centering, adaptive weights and initialization are recomputed on each
training fold. Errors are mean squared errors per response, weighted by fold
size. `cvm`, `cvse`, `folds`, `path` and `index` (minimum-error solution) are
returned. `coef` and `predict` use that solution. A failed fold solve is an
error; it is never silently selected. For rank paths choose ranks estimable
in every training fold (an invalid rank raises an error).
"""
struct CVLowRankPath{T<:AbstractFloat}
  path::LowRankPath{T}
  cvm::Vector{T}
  cvse::Vector{T}
  folds::Vector{Int}
  index::Int
end
function CVLowRankPath(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                       nfolds::Integer = 5, folds = nothing, rng::AbstractRNG = default_rng(), kwargs...)
  n = size(X, 1)
  fl, K = _foldlabels(rng, n, nfolds, folds)
  p = LowRankPath(X, Y; kwargs...)
  all(s -> s.converged, p.solutions) || throw(ErrorException("full-data low-rank path did not converge"))
  T = eltype(p.λpath)
  errors = zeros(T, K, length(p.solutions))
  counts = [count(==(k), fl) for k in 1:K]
  for k in 1:K
    tr, te = findall(!=(k), fl), findall(==(k), fl)
    grid = p.method in (:rank, :sparse_svd) ? (; ranks = p.ranks) : (; λpath = p.λpath)
    opts = merge((; kwargs...), grid)
    fp = LowRankPath(X[tr, :], Y[tr, :]; opts...)
    for (j, s) in enumerate(fp.solutions)
      s.converged || throw(ErrorException("low-rank fit did not converge in fold $k, level $j"))
      errors[k, j] = sum(abs2, Y[te, :] - predict(s, X[te, :])) / (length(te) * size(Y, 2))
    end
  end
  cvm = vec(sum(errors .* counts; dims = 1)) / n
  cvse = [sqrt(sum(counts .* (errors[:, j] .- cvm[j]).^2) / (n*(K-1))) for j in eachindex(cvm)]
  CVLowRankPath(p, cvm, cvse, fl, argmin(cvm))
end

StatsAPI.coef(s::LowRankSolution) = s.hasintercept ? [s.intercept'; s.B] : copy(s.B)
function StatsAPI.predict(s::LowRankSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, size(s.B, 1))
  X * s.B .+ s.intercept'
end
StatsAPI.residuals(s::LowRankSolution) = s.residuals
StatsAPI.nobs(s::LowRankSolution) = size(s.residuals, 1)
StatsAPI.dof(s::LowRankSolution) = s.df
StatsAPI.deviance(s::LowRankSolution) = sum(abs2, s.residuals)
StatsAPI.aic(s::LowRankSolution) = length(s.residuals) * (lowrank_ic(s; criterion = :aic) - log(length(s.residuals)))
StatsAPI.bic(s::LowRankSolution) = length(s.residuals) * (lowrank_ic(s; criterion = :bic) - log(length(s.residuals)))
StatsAPI.coef(p::LowRankPath) = coef.(p.solutions)
StatsAPI.predict(p::LowRankPath, X::AbstractMatrix) = [predict(s, X) for s in p.solutions]
StatsAPI.coef(cv::CVLowRankPath) = coef(cv.path.solutions[cv.index])
StatsAPI.predict(cv::CVLowRankPath, X::AbstractMatrix) = predict(cv.path.solutions[cv.index], X)
StatsAPI.fit(::Type{LowRankPath}, X::AbstractMatrix, Y::AbstractMatrix; kwargs...) = LowRankPath(X, Y; kwargs...)
StatsAPI.fit(::Type{CVLowRankPath}, X::AbstractMatrix, Y::AbstractMatrix; kwargs...) = CVLowRankPath(X, Y; kwargs...)
