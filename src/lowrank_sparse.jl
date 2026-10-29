######################################################################
# Simultaneously sparse and low-rank regression
######################################################################

"""
    SparseNuclearSolution

A fit of `‖Y - 1β₀' - XB‖²_F/(2n) + λ_rank ‖B‖_* + λ_sparse P(B)`.
`B`, `intercept`, `residuals`, `objective`, `activerows` and `nnz` describe
the returned coefficients. `weights` are positive row weights; `sparsity`
is `:entry` or `:row`. `rank` counts singular values above the reported
`rank_tol`, which also accounts for disagreement with the nuclear proximal
point. `gap` is the relative duality gap, `NaN` for unpenalized least squares.
`iterations` and `converged` report solver termination; `r_norm`, `s_norm`,
`eps_pri` and `eps_dual` retain its KKT residuals and tolerances.
`coef`, `predict`, `residuals`, `nobs` and `deviance` apply. No degrees of
freedom or information criterion is provided for this estimator.
"""
struct SparseNuclearSolution{T<:AbstractFloat}
  B::Matrix{T}
  intercept::Vector{T}
  λ_rank::T
  λ_sparse::T
  sparsity::Symbol
  weights::Vector{T}
  rank::Int
  rank_tol::T
  activerows::Vector{Int}
  nnz::Int
  objective::T
  gap::T
  iterations::Int
  converged::Bool
  residuals::Matrix{T}
  hasintercept::Bool
  r_norm::T
  s_norm::T
  eps_pri::T
  eps_dual::T
end

struct _SparseNuclearLoss{D,G,T} <: ProximalBase.AbstractFunction
  data::D
  setup::G
  work::Matrix{T}
end
ProximalBase.isconvex(::Type{<:_SparseNuclearLoss}) = true
ProximalBase.issmooth(::Type{<:_SparseNuclearLoss}) = true
ProximalBase.lipschitz(f::_SparseNuclearLoss) = f.setup.L
function (f::_SparseNuclearLoss)(x::AbstractVector)
  d = f.data
  mul!(f.work, d.X, reshape(x, size(f.setup.C)))
  f.work .-= d.Y
  sum(abs2, f.work) / (2size(d.X, 1))
end
function ProximalBase.gradient!(out::AbstractVector, f::_SparseNuclearLoss, x::AbstractVector)
  d, g = f.data, f.setup
  B, G = reshape(x, size(g.C)), reshape(out, size(g.C))
  # Form residuals before the transpose product, as in the gap certificate.
  mul!(f.work, d.X, B)
  f.work .-= d.Y
  mul!(G, d.X', f.work)
  G ./= size(d.X, 1)
  out
end
function ProximalBase.value_and_gradient!(out, f::_SparseNuclearLoss, x)
  ProximalBase.gradient!(out, f, x)
  f(x)
end

# Conjugate uses this prox through Moreau's identity, including in-place calls.
struct _SparseNuclearShape{F,T} <: ProximalBase.AbstractFunction
  f::F
  point::Matrix{T}
end
ProximalBase.isconvex(::Type{<:_SparseNuclearShape{F}}) where {F} = ProximalBase.isconvex(F)
ProximalBase.isproximable(::Type{<:_SparseNuclearShape{F}}) where {F} = ProximalBase.isproximable(F)
(h::_SparseNuclearShape)(x::AbstractVector) = h.f(reshape(x, size(h.point)))
function ProximalBase.prox!(out::AbstractVector, h::_SparseNuclearShape, x::AbstractVector, γ::Real)
  prox!(reshape(out, size(h.point)), h.f, reshape(x, size(h.point)), γ)
  copyto!(h.point, reshape(out, size(h.point)))
  out
end

struct _SparseNuclearProblem{D,F,T}
  data::D
  loss::F
  weights::Vector{T}
  sparsity::Symbol
  max_iter::Int
  tol::T
  λmax::NTuple{2,T}
  names::NTuple{2,Symbol}
end

_sn_penalty(B, w, sparsity) = sum(w[j] * (sparsity == :entry ?
  sum(abs, view(B, j, :)) : norm(view(B, j, :))) for j in axes(B, 1))
_sn_dualnorm(V, w, sparsity) = maximum((sparsity == :entry ?
  maximum(abs, view(V, j, :)) : norm(view(V, j, :))) / w[j] for j in axes(V, 1))

function sparse_nuclear_regression end

function _grid_problem(::typeof(sparse_nuclear_regression), X::AbstractMatrix{<:Real},
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
  # Residual gradients need only the cross-product and design spectral scale.
  n = size(d.X, 1)
  g = (; C = d.X' * d.Y / n, L = opnorm(d.X)^2 / n)
  f = _SparseNuclearLoss(d, g, similar(d.Y))
  λmax = (opnorm(g.C), _sn_dualnorm(g.C, w, sparsity))
  _SparseNuclearProblem(d, f, w, sparsity, Int(max_iter), T(tol), λmax, (:rank, :sparse))
end

function _sn_relative_gap(primal::T, dual::T) where {T}
  isfinite(primal) && isfinite(dual) || return T(Inf)
  δ = primal - dual
  δ < -100eps(T) * max(abs(primal), abs(dual), floatmin(T)) && return T(Inf)
  max(zero(T), δ) / max(primal, floatmin(T))
end

function _sn_gap(problem, B, U, λr, λs)
  T = eltype(B)
  iszero(λr) && iszero(λs) && return T(NaN)
  all(isfinite, B) && all(isfinite, U) || return T(Inf)
  d, w, sparsity = problem.data, problem.weights, problem.sparsity
  n = size(d.X, 1)
  R = d.Y - d.X * B
  C = d.X' * R / n
  all(isfinite, R) && all(isfinite, C) || return T(Inf)
  primal = sum(abs2, R)/(2n) + λr*sum(svdvals(B)) + λs*_sn_penalty(B, w, sparsity)
  if iszero(λs)
    scale = opnorm(C)
    t = iszero(scale) ? one(T) : min(one(T), λr/scale)
  else
    V = iszero(λr) ? C : C - U
    scale = _sn_dualnorm(V, w, sparsity)
    t = iszero(scale) ? one(T) : min(one(T), λs/scale)
    # The dual prox enforces this ball; retain feasibility under roundoff too.
    if !iszero(λr)
      nu = opnorm(U)
      iszero(nu) || (t = min(t, λr/nu))
    end
  end
  dual = t*dot(R, d.Y)/n - t^2*sum(abs2, R)/(2n)
  _sn_relative_gap(primal, dual)
end

function _sn_solution(problem, B, Bh, λr, λs, gap, result)
  d, w = problem.data, problem.weights
  T = eltype(B)
  σ = svdvals(B)
  rank_tol = max(sqrt(eps(T))*first(σ), opnorm(B - Bh))
  E = d.Y - d.X * B
  objective = sum(abs2, E)/(2size(d.X, 1)) + λr*sum(σ) + λs*_sn_penalty(B, w, problem.sparsity)
  converged = result.converged && (iszero(λr) && iszero(λs) || gap <= problem.tol)
  SparseNuclearSolution(B, d.my - B' * d.mx, λr, λs, problem.sparsity,
    copy(w), count(>(rank_tol), σ), rank_tol, _activerows(B), count(!iszero, B),
    objective, gap, result.iterations, converged, E, d.intercept,
    result.r_norm, result.s_norm, result.eps_pri, result.eps_dual)
end

function _grid_fit(problem::_SparseNuclearProblem, λ_rank, λ_sparse, warm)
  _lr_options(λ_rank, problem.max_iter, problem.tol)
  _lr_options(λ_sparse, problem.max_iter, problem.tol)
  d, f = problem.data, problem.loss
  T = eltype(d.X)
  λr, λs = T(λ_rank), T(λ_sparse)
  _lr_options(λr, problem.max_iter, problem.tol)
  _lr_options(λs, problem.max_iter, problem.tol)
  p, q = size(f.setup.C)
  if iszero(f.setup.L)
    B = zeros(T, p, q)
    result = (; iterations = 0, converged = true, r_norm = zero(T), s_norm = zero(T),
      eps_pri = zero(T), eps_dual = zero(T))
    gap = _sn_gap(problem, B, B, λr, λs)
    solution = _sn_solution(problem, B, B, λr, λs, gap, result)
    return solution, (; x = vec(copy(B)), y = zeros(T, p*q), τ = nothing, σ = nothing)
  end
  x = warm === nothing ? zeros(T, p*q) : copy(warm.x)
  y = warm === nothing ? zeros(T, p*q) : copy(warm.y)
  g = problem.sparsity == :entry ? NormL1(λs, repeat(problem.weights, q)) :
    GroupNormL2(λs, [j:p:p*q for j in 1:p], problem.weights)
  h = _SparseNuclearShape(NuclearNorm(λr), zeros(T, p, q))
  steps = warm === nothing ? (;) : (; τ = warm.τ, σ = warm.σ)
  # Without averaging, the last nuclear proximal point belongs to this pair.
  # A zero nuclear term has no dual scale to balance against the primal one.
  method = CondatVu(; L = f.setup.L, norm_d = one(T), abstol = problem.tol,
    reltol = problem.tol, restart = false, adaptive = !iszero(λr), steps...)
  penalized = !iszero(λr) || !iszero(λs)
  gapfun = (x, y) -> _sn_gap(problem, reshape(x, p, q), reshape(y, p, q), λr, λs)
  # ProximalOptions requires maxiter > 1; a callback permits a single step.
  callback = problem.max_iter == 1 ? _ -> true : nothing
  options = ProximalOptions(; maxiter = max(2, problem.max_iter), callback,
    gaptol = penalized ? problem.tol : 0)
  result = solve!(x, f, g, h, Diagonal(ones(T, p*q)), method, options; y, gap = gapfun)
  B = copy(reshape(x, p, q))
  gap = penalized ? T(gapfun(x, y)) : T(NaN)
  solution = _sn_solution(problem, B, h.point, λr, λs, gap, result)
  solution, (; x = copy(x), y = copy(y), τ = result.τ, σ = result.σ)
end

"""
    sparse_nuclear_regression(X, Y, λ_rank, λ_sparse; sparsity = :entry,
                              weights = nothing, intercept = true,
                              max_iter = 10_000, tol = 1e-7)

Minimize `‖Y - 1β₀' - XB‖²_F/(2n) + λ_rank ‖B‖_* + λ_sparse P(B)`
by ProximalSolvers' Condat–Vũ splitting. Both penalties act on the same `B`.
For `sparsity = :entry`, `P(B) = ∑ⱼ weights[j] ∑ₜ |B[j,t]|`; for `:row`,
`P(B) = ∑ⱼ weights[j] ‖B[j,:]‖₂`. Row weights are positive and finite,
one by default. Levels are finite and nonnegative. Columns keep their
original scales, and intercepts are unpenalized.

Returns a [`SparseNuclearSolution`](@ref). `tol` bounds its relative duality
gap at the returned coefficients. With both levels zero, the solver's KKT
residual rule is used and `gap = NaN`. A constant centered design gives the
intercept-only fit. See [`PenaltyGrid`](@ref) for warm starts and
[`CVPenaltyGrid`](@ref) for tuning both levels.
"""
function sparse_nuclear_regression(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real},
                                   λ_rank::Real, λ_sparse::Real;
                                   sparsity::Symbol = :entry, weights = nothing,
                                   intercept::Bool = true, max_iter::Integer = 10_000,
                                   tol::Real = 1e-7)
  _lr_options(λ_rank, max_iter, tol); _lr_options(λ_sparse, max_iter, tol)
  problem = _grid_problem(sparse_nuclear_regression, X, Y; sparsity, weights, intercept, max_iter, tol)
  first(_grid_fit(problem, λ_rank, λ_sparse, nothing))
end

StatsAPI.coef(s::SparseNuclearSolution) = s.hasintercept ? [s.intercept'; s.B] : copy(s.B)
function StatsAPI.predict(s::SparseNuclearSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, size(s.B, 1))
  X * s.B .+ s.intercept'
end
StatsAPI.residuals(s::SparseNuclearSolution) = s.residuals
StatsAPI.nobs(s::SparseNuclearSolution) = size(s.residuals, 1)
StatsAPI.deviance(s::SparseNuclearSolution) = sum(abs2, s.residuals)
