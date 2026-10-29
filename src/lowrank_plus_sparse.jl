######################################################################
# Low-rank-plus-sparse regression
######################################################################

"""
    LowRankPlusSparseSolution

A fit of `‖Y - 1β₀' - X(L + S)‖²_F/(2n) + λ_rank ‖L‖_* + λ_sparse P(S)`.
`L`, `S` and `B = L + S` are independent matrices. `intercept`, `residuals`,
`objective`, `coef` and `predict` use the total `B`. `weights` are positive
predictor weights, and `sparsity` is `:entry` or `:row`.

`rank` counts singular values of `B` above `rank_tol = √eps(T) σ₁(B)`.
`lowrank_rank` is the rank kept by the last nuclear proximal step (the numerical
rank of `L` on a zero-penalty axis). `sparse_nnz` and `sparse_rows` describe the
exact support of `S`. `gap` is the relative intersection-dual gap, up to
floating-point accuracy. `lowrank_residual` and `sparse_residual` are Frobenius norms
of the component proximal-gradient mappings at the returned pair.
`iterations` and `converged` describe termination. On either zero-penalty
axis, a minimum-norm least-squares convention applies and `gap = NaN`.

Fitted values are unique; components need not be. A small gap bounds
objective error, not component error. `coef`, `predict`, `residuals`, `nobs`
and `deviance` apply. No degrees of freedom or information criterion is
provided for this estimator.
"""
struct LowRankPlusSparseSolution{T<:AbstractFloat}
  L::Matrix{T}
  S::Matrix{T}
  B::Matrix{T}
  intercept::Vector{T}
  λ_rank::T
  λ_sparse::T
  sparsity::Symbol
  weights::Vector{T}
  rank::Int
  rank_tol::T
  lowrank_rank::Int
  sparse_nnz::Int
  sparse_rows::Vector{Int}
  objective::T
  gap::T
  lowrank_residual::T
  sparse_residual::T
  iterations::Int
  converged::Bool
  residuals::Matrix{T}
  hasintercept::Bool
end

struct _LowRankPlusSparseProblem{D,G,T}
  data::D
  setup::G
  weights::Vector{T}
  sparsity::Symbol
  max_iter::Int
  tol::T
  λmax::NTuple{2,T}
  names::NTuple{2,Symbol}
end

function lowrank_plus_sparse end

function _grid_problem(::typeof(lowrank_plus_sparse), X::AbstractMatrix{<:Real},
                       Y::AbstractMatrix{<:Real}; sparsity::Symbol = :entry,
                       weights = nothing, intercept::Bool = true,
                       max_iter::Integer = 10_000, tol::Real = 1e-7,
                       rng::AbstractRNG = default_rng())
  _lr_options(0, max_iter, tol)
  sparsity in (:entry, :row) || throw(ArgumentError("sparsity must be :entry or :row"))
  d = _lr_data(X, Y, intercept)
  T = eltype(d.X)
  _lr_options(0, max_iter, T(tol))
  w = collect(_lr_weights(d, nothing, weights, false, 2))
  g = _lr_gradient(d)
  λmax = (opnorm(g.C), _sn_dualnorm(g.C, w, sparsity))
  _LowRankPlusSparseProblem(d, g, w, sparsity, Int(max_iter), T(tol), λmax, (:rank, :sparse))
end

function _lps_sparse_penalty(problem, λ)
  p, q = size(problem.setup.C)
  problem.sparsity == :entry ? NormL1(λ, repeat(problem.weights, q)) :
    GroupNormL2(λ, [j:p:p*q for j in 1:p], problem.weights)
end

function _lps_certificate(problem, L, S, λr, λs)
  d = problem.data
  T, n = eltype(L), size(d.X, 1)
  R = d.Y - d.X * (L + S)
  C = d.X' * R / n
  if !all(isfinite, L) || !all(isfinite, S) || !all(isfinite, R) || !all(isfinite, C)
    return (; objective = T(Inf), gap = T(Inf), R, C)
  end
  objective = sum(abs2, R)/(2n) + λr*sum(svdvals(L)) +
    λs*_sn_penalty(S, problem.weights, problem.sparsity)
  if iszero(λr) || iszero(λs)
    return (; objective, gap = T(NaN), R, C)
  end
  nr, ns = opnorm(C), _sn_dualnorm(C, problem.weights, problem.sparsity)
  t = one(T)
  iszero(nr) || (t = min(t, λr/nr))
  iszero(ns) || (t = min(t, λs/ns))
  dual = t*dot(R, d.Y)/n - t^2*sum(abs2, R)/(2n)
  (; objective, gap = _sn_relative_gap(objective, dual), R, C)
end

function _lps_solution(problem, L, S, λr, λs, iterations; lowrank_rank = nothing)
  d = problem.data
  T = eltype(L)
  B = L + S
  σ = svdvals(B)
  rank_tol = sqrt(eps(T))*first(σ)
  if lowrank_rank === nothing
    σL = svdvals(L)
    lowrank_rank = count(>(sqrt(eps(T))*first(σL)), σL)
  end
  cert = _lps_certificate(problem, L, S, λr, λs)
  γ = iszero(problem.setup.L) ? one(T) : inv(2problem.setup.L)
  PL, PS = similar(L), similar(S)
  prox!(PL, NuclearNorm(λr), L + γ*cert.C, γ)
  prox!(PS, _lps_sparse_penalty(problem, λs), S + γ*cert.C, γ)
  lowrank_residual, sparse_residual = norm(L - PL)/γ, norm(S - PS)/γ
  # Axes use the SVD least-squares solution and its normal equations.
  converged = iszero(λr) || iszero(λs) ?
    norm(cert.C) <= problem.tol*max(one(T), norm(problem.setup.C)) : cert.gap <= problem.tol
  LowRankPlusSparseSolution(copy(L), copy(S), B, d.my - B' * d.mx,
    λr, λs, problem.sparsity, copy(problem.weights), count(>(rank_tol), σ),
    rank_tol, lowrank_rank, count(!iszero, S), _activerows(S), cert.objective,
    cert.gap, lowrank_residual, sparse_residual, iterations, converged, cert.R, d.intercept)
end

function _grid_fit(problem::_LowRankPlusSparseProblem, λ_rank, λ_sparse, warm)
  _lr_options(λ_rank, problem.max_iter, problem.tol)
  _lr_options(λ_sparse, problem.max_iter, problem.tol)
  d, g = problem.data, problem.setup
  T = eltype(d.X)
  λr, λs = T(λ_rank), T(λ_sparse)
  _lr_options(λr, problem.max_iter, problem.tol)
  _lr_options(λs, problem.max_iter, problem.tol)
  p, q = size(g.C)
  L, S = zeros(T, p, q), zeros(T, p, q)
  if iszero(λr) || iszero(λs)
    f = _lr_svd(d)
    B = f.V * ((f.U' * d.Y) ./ f.s)
    iszero(λr) ? copyto!(L, B) : copyto!(S, B)
    solution = _lps_solution(problem, L, S, λr, λs, 0)
    return solution, (; L = copy(L), S = copy(S))
  end
  if iszero(g.L) || (λr >= problem.λmax[1] && λs >= problem.λmax[2])
    solution = _lps_solution(problem, L, S, λr, λs, 0; lowrank_rank = 0)
    return solution, (; L = copy(L), S = copy(S))
  end
  if warm !== nothing
    copyto!(L, warm.L)
    copyto!(S, warm.S)
  end
  ZL, ZS = copy(L), copy(S)
  candidate_L, candidate_S = similar(L), similar(S)
  step_L, step_S = similar(L), similar(S)
  γ = inv(2g.L)
  nuclear, sparse = NuclearNorm(λr), _lps_sparse_penalty(problem, λs)
  t, iteration = one(T), 0
  for k in 1:problem.max_iter
    iteration = k
    G = _lr_grad(d, g, ZL + ZS)
    @. step_L = ZL - γ * G
    @. step_S = ZS - γ * G
    prox!(candidate_L, nuclear, step_L, γ)
    prox!(candidate_S, sparse, step_S, γ)
    if k == 1 || k % 5 == 0 || k == problem.max_iter
      cert = _lps_certificate(problem, candidate_L, candidate_S, λr, λs)
      if cert.gap <= problem.tol
        L .= candidate_L
        S .= candidate_S
        break
      end
    end
    tn = (one(T) + sqrt(one(T) + 4t^2)) / 2
    β = (t - 1) / tn
    if dot(ZL - candidate_L, candidate_L - L) + dot(ZS - candidate_S, candidate_S - S) > 0
      tn, β = one(T), zero(T)
    end
    @. ZL = candidate_L + β * (candidate_L - L)
    @. ZS = candidate_S + β * (candidate_S - S)
    L .= candidate_L
    S .= candidate_S
    t = tn
  end
  lowrank_rank = count(>(γ*λr), svdvals(step_L))
  solution = _lps_solution(problem, L, S, λr, λs, iteration; lowrank_rank)
  solution, (; L = copy(L), S = copy(S))
end

"""
    lowrank_plus_sparse(X, Y, λ_rank, λ_sparse; sparsity = :entry,
                        weights = nothing, intercept = true,
                        max_iter = 10_000, tol = 1e-7)

Minimize `‖Y - 1β₀' - X(L + S)‖²_F/(2n) + λ_rank ‖L‖_* + λ_sparse P(S)`
by restarted accelerated proximal gradient on `(L, S)`, with step
`1/(2‖Xc‖²_op/n)` and ProximalBase's nuclear and sparsity proximal maps.
For `sparsity = :entry`, `P(S) = ∑ⱼ weights[j] ∑ₜ |S[j,t]|`; for `:row`,
`P(S) = ∑ⱼ weights[j] ‖S[j,:]‖₂`. Weights are positive finite predictor
weights, one by default. Observation, task and domain weights are unsupported.
Levels are finite and nonnegative. Intercepts are unpenalized, columns keep
their original scales, and the loss sums over responses without dividing by
their number. Only a shared design matrix is supported.

Returns a [`LowRankPlusSparseSolution`](@ref). With positive levels, `tol`
bounds the relative intersection-dual gap at the returned pair, up to
floating-point accuracy. A small gap does not bound component error. If
`λ_rank = 0`, the minimum-norm least-squares coefficient is assigned to `L`
and `S = 0`, including when both levels are zero. If only `λ_sparse = 0`,
it is assigned to `S` and `L = 0`. These axes have `gap = NaN` and use the
least-squares normal equations. A zero centered design gives an intercept-only
fit. Components need not be unique, and no identifiability constraints are
imposed. See [`PenaltyGrid`](@ref) for warm starts and [`CVPenaltyGrid`](@ref)
for tuning both levels; automatic axes use the full-data zero-pair references.
"""
function lowrank_plus_sparse(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real},
                             λ_rank::Real, λ_sparse::Real;
                             sparsity::Symbol = :entry, weights = nothing,
                             intercept::Bool = true, max_iter::Integer = 10_000,
                             tol::Real = 1e-7)
  _lr_options(λ_rank, max_iter, tol); _lr_options(λ_sparse, max_iter, tol)
  problem = _grid_problem(lowrank_plus_sparse, X, Y; sparsity, weights, intercept, max_iter, tol)
  first(_grid_fit(problem, λ_rank, λ_sparse, nothing))
end

function _grid_folds(::typeof(lowrank_plus_sparse), X, Y; nfolds, folds, rng)
  size(X, 1) > 0 || throw(ArgumentError("X must have nonempty dimensions"))
  fl, K = _foldlabels(rng, size(X, 1), nfolds, folds)
  all(>(0), fl) || throw(ArgumentError("fold labels must be positive"))
  fl, K
end
function _grid_train(::typeof(lowrank_plus_sparse), X, Y, folds, k)
  tr = findall(!=(k), folds)
  (X[tr, :], Y[tr, :])
end
function _grid_score(::typeof(lowrank_plus_sparse), solution, X, Y, folds, k)
  te = findall(==(k), folds)
  (sum(abs2, Y[te, :] - predict(solution, X[te, :])) / size(Y, 2), length(te))
end

StatsAPI.coef(s::LowRankPlusSparseSolution) = s.hasintercept ? [s.intercept'; s.B] : copy(s.B)
function StatsAPI.predict(s::LowRankPlusSparseSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, size(s.B, 1))
  X * s.B .+ s.intercept'
end
StatsAPI.residuals(s::LowRankPlusSparseSolution) = s.residuals
StatsAPI.nobs(s::LowRankPlusSparseSolution) = size(s.residuals, 1)
StatsAPI.deviance(s::LowRankPlusSparseSolution) = sum(abs2, s.residuals)
