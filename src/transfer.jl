######################################################################
#
#   Joint-sparse transfer multitask learning
#
######################################################################

_domain_name(l) = l == 0 ? "the target" : "source $l"

# A domain (X, Y) as K designs, one per response (the same object when X is
# one matrix), and the n × K responses.
function _transfer_domain(domain, name::AbstractString)
  domain isa Union{Tuple, Pair, AbstractVector} && length(domain) == 2 ||
    throw(ArgumentError("$name must be a pair (X, Y) of designs and responses."))
  X, Y = domain
  Y isa AbstractVecOrMat || throw(ArgumentError("the responses of $name must be a vector or a matrix."))
  Ym = Y isa AbstractVector ? reshape(Y, :, 1) : Y
  K = size(Ym, 2)
  K >= 1 || throw(ArgumentError("$name must have at least one response."))
  X isa AbstractMatrix && return fill(X, K), Ym
  X isa AbstractVector && !isempty(X) && all(x -> x isa AbstractMatrix, X) ||
    throw(ArgumentError("the design of $name must be a matrix or a vector of matrices, one per response."))
  length(X) == K || throw(DimensionMismatch("$name has $(length(X)) designs for $K responses"))
  collect(X), Ym
end

# The rows of a domain, keeping a design shared by its responses.
function _transfer_rows((Xs, Y), rows)
  Xr = Vector{AbstractMatrix}(undef, length(Xs))
  for k in eachindex(Xs)
    u = findfirst(s -> Xs[s] === Xs[k], 1:k-1)
    Xr[k] = u === nothing ? Xs[k][rows, :] : Xr[u]
  end
  map(identity, Xr), Y[rows, :]
end

# The domains ℓ = 0, …, L, the target first. Task t = ℓK + k is response k of
# domain ℓ, with design X[t]; a domain with one design repeats it. With an
# intercept, columns and responses are centered within each task.
struct _TransferData{T<:AbstractFloat, VX<:AbstractVector{<:AbstractMatrix{T}}}
  X::VX
  y::Vector{Vector{T}}
  nobs::Vector{Int}                          # n₀, …, n_L
  N::Int                                     # ∑ₗ nₗ
  K::Int
  intercept::Bool
  shared::Vector{Bool}                       # one design for all responses of domain ℓ
  xbar::Matrix{T}                            # p × (L + 1)K, zero without an intercept
  ybar::Vector{T}
  norms::Matrix{T}                           # ‖x̃ₜⱼ‖², the squared norms of the centered columns
end

_TransferData(target, sources::AbstractVector; intercept::Bool) =
  _TransferData([_transfer_domain(target, "the target");
                 [_transfer_domain(s, _domain_name(l)) for (l, s) in enumerate(sources)]]; intercept)

function _TransferData(domains::AbstractVector; intercept::Bool)
  K, p = size(domains[1][2], 2), size(first(domains[1][1]), 2)
  n = Int[]
  for (i, (Xs, Y)) in enumerate(domains)
    name = _domain_name(i - 1)
    size(Y, 2) == K || throw(DimensionMismatch("$name has $(size(Y, 2)) responses, but the target has $K"))
    m = size(Y, 1)
    for X in Xs
      size(X, 2) == p || throw(DimensionMismatch("a design of $name has $(size(X, 2)) columns, but the target's have $p"))
      size(X, 1) == m || throw(DimensionMismatch("a design of $name has $(size(X, 1)) rows, but its responses have $m"))
    end
    m >= 1 + intercept ||
      throw(ArgumentError("$name has $m observations; a fit$(intercept ? " with an intercept" : "") needs at least $(1 + intercept)."))
    eltype(Y) <: Real && all(X -> eltype(X) <: Real, Xs) ||
      throw(ArgumentError("the designs and the responses of $name must have real entries."))
    _check_finite(Y, "the responses of $name")
    foreach(X -> _check_finite(X isa SparseMatrixCSC ? nonzeros(X) : X, "a design of $name"), Xs)
    push!(n, m)
  end
  T = float(mapreduce(((Xs, Y),) -> promote_type(eltype(Y), mapreduce(eltype, promote_type, Xs)), promote_type, domains))
  raw = [Xs[k] for (Xs, _) in domains for k in 1:K]
  q = length(raw)
  Xv = Vector{AbstractMatrix{T}}(undef, q)
  for t in 1:q
    u = findfirst(s -> raw[s] === raw[t], 1:t-1)
    Xv[t] = u === nothing ? _design(convert(AbstractMatrix{T}, raw[t])) : Xv[u]
  end
  X = map(identity, Xv)
  y = [convert(Vector{T}, Y[:, k]) for (_, Y) in domains for k in 1:K]
  xbar, ybar, norms = zeros(T, p, q), zeros(T, q), zeros(T, p, q)
  for t in 1:q
    u = findfirst(s -> X[s] === X[t], 1:t-1)
    if u === nothing
      intercept && (xbar[:, t] = _column_means(X[t]))
      for j in 1:p
        norms[j, t] = _row_norm2(X[t], j, xbar[j, t])
      end
    else
      xbar[:, t] = view(xbar, :, u); norms[:, t] = view(norms, :, u)
    end
    intercept && (ybar[t] = mean(y[t]))
  end
  shared = [all(k -> X[l * K + k] === X[l * K + 1], 1:K) for l in 0:length(n)-1]
  _TransferData{T, typeof(X)}(X, y, n, sum(n), K, intercept, shared, xbar, ybar, norms)
end

# The contrast weights a₁, …, a_L: 8√(nₗ/N), 8nₗ/N or the positive values given.
function _contrast_weights(::Type{T}, weights, n::Vector{Int}) where {T}
  L, N = length(n) - 1, sum(n)
  weights === :sqrt && return T[8 * sqrt(T(n[l + 1]) / N) for l in 1:L]
  weights === :linear && return T[8 * T(n[l + 1]) / N for l in 1:L]
  weights isa AbstractVector{<:Real} ||
    throw(ArgumentError("contrast_weights must be :sqrt, :linear or a vector of positive weights, but it is $(repr(weights))."))
  length(weights) == L || throw(DimensionMismatch("there are $L sources and $(length(weights)) contrast weights"))
  all(a -> isfinite(a) && a > 0, weights) || throw(ArgumentError("the contrast weights must be positive and finite."))
  Vector{T}(weights)
end

# Coordinate descent on θ⁰ = B⁰ and θˡ = Bˡ - B⁰. Column k of θ⁰ enters
# response k of every domain, and column k of θˡ that of domain ℓ only. Block
# b = (j - 1)(L + 1) + c + 1 is row j of component c.
struct _TransferState{T, D<:_TransferData{T}}
  data::D
  θ::Vector{Matrix{T}}                       # θ⁰, θ¹, …, θᴸ, each p × K
  residuals::Vector{Vector{T}}               # ỹₜ - X̃ₜ Bˡ[:, k]
  weights::Vector{T}                         # 1, a₁, …, a_L
  penalties::Vector{_RowPenalty{T}}          # row ℓ2 norms with these weights
  coordinate_curvature::Array{T, 3}          # p × K × (L + 1)
  curvature::Matrix{T}                       # p × (L + 1), the row steps H = maxₖ h
  response_scale::T
end

function _TransferState(d::_TransferData{T}, a::Vector{T}) where {T}
  p, K, L = size(d.xbar, 1), d.K, length(d.nobs) - 1
  NK = T(d.N * K)
  h = zeros(T, p, K, L + 1)
  for k in 1:K, j in 1:p
    h[j, k, 1] = sum(d.norms[j, l * K + k] for l in 0:L) / NK
    for l in 1:L
      h[j, k, l + 1] = d.norms[j, l * K + k] / NK
    end
  end
  residuals = [d.y[t] .- d.ybar[t] for t in eachindex(d.y)]
  w = [one(T); a]
  _TransferState(d, [zeros(T, p, K) for _ in 0:L], residuals, w,
                 [_RowPenalty(:l2, zero(T), fill(wc, p)) for wc in w], h,
                 dropdims(maximum(h; dims = 2); dims = 2), sqrt(sum(r -> sum(abs2, r), residuals) / NK))
end

_transfer_block(b, C) = (b - 1) % C, (b - 1) ÷ C + 1
_transfer_domains(c, L) = c == 0 ? (0:L) : (c:c)

# The gradient of block (c, j): minus the scores Sˡⱼₖ = ⟨x̃ₜⱼ, rₜ⟩/(NK) summed
# over the domains the component enters.
function _transfer_gradient!(g, st::_TransferState{T}, c, j) where {T}
  d = st.data
  K, L = d.K, length(d.nobs) - 1
  NK = T(d.N * K)
  for k in 1:K
    s = zero(T)
    for l in _transfer_domains(c, L)
      t = l * K + k
      iszero(d.norms[j, t]) || (s += _row_dot(d.X[t], st.residuals[t], j, d.xbar[j, t]))
    end
    g[k] = -s / NK
  end
  g
end

function _transfer_residuals!(st::_TransferState, c, j, k, delta)
  d = st.data
  for l in _transfer_domains(c, length(d.nobs) - 1)
    t = l * d.K + k
    iszero(d.norms[j, t]) || _row_residual!(st.residuals[t], d.X[t], j, d.xbar[j, t], delta)
  end
end

# ‖gⱼ‖₂ / w of every block, p × (L + 1), from one pass over the scores.
function _transfer_duals(st::_TransferState{T}) where {T}
  d = st.data
  p, K, L = size(d.xbar, 1), d.K, length(d.nobs) - 1
  NK = T(d.N * K)
  S = zeros(T, p, K, L + 1)
  for l in 0:L, k in 1:K
    t = l * K + k
    for j in 1:p
      iszero(d.norms[j, t]) || (S[j, k, l + 1] = _row_dot(d.X[t], st.residuals[t], j, d.xbar[j, t]) / NK)
    end
  end
  M = zeros(T, p, L + 1)
  for j in 1:p
    M[j, 1] = sqrt(sum(k -> abs2(sum(view(S, j, k, :))), 1:K)) / st.weights[1]
    for l in 1:L
      M[j, l + 1] = norm(view(S, j, :, l + 1)) / st.weights[l + 1]
    end
  end
  M
end

_transfer_active(st::_TransferState, b) =
  ((c, j) = _transfer_block(b, length(st.θ)); any(!iszero, view(st.θ[c + 1], j, :)))

# The global relative duality gap on the candidate blocks, at the dual point
# u = s r/(NK) with s = min(1, λ/M); excluded blocks are fixed at zero.
function _transfer_gap(st::_TransferState{T}, λ, blocks) where {T}
  iszero(λ) && return nothing
  d, C = st.data, length(st.θ)
  NK = T(d.N * d.K)
  P = sum(r -> sum(abs2, r), st.residuals) / (2NK)
  P += λ * sum(blocks; init = zero(T)) do b
    c, j = _transfer_block(b, C)
    _row_value(view(st.θ[c + 1], j, :), st.penalties[c + 1], j)
  end
  iszero(P) && return nothing
  M = _transfer_duals(st)
  scale = one(T)
  for b in blocks
    c, j = _transfer_block(b, C)
    M[j, c + 1] > λ && (scale = min(scale, λ / M[j, c + 1]))
  end
  D = zero(T)
  for t in eachindex(d.y)
    r = st.residuals[t]
    yr = sum(i -> (d.y[t][i] - d.ybar[t]) * r[i], eachindex(r))
    D += (scale * yr - scale^2 * sum(abs2, r) / 2) / NK
  end
  isfinite(P) && isfinite(D) || return T(Inf)
  gap = P - D
  gap >= -8 * eps(T) * max(abs(P), abs(D)) || return T(Inf)
  max(gap, 4 * eps(T) * abs(P)) / max(abs(P), floatmin(T))
end

# Full passes alternate with passes on the active blocks, as in _row_solve!.
# The row step uses H = maxₖ h, a majorization when the curvatures differ.
function _transfer_solve!(st::_TransferState{T}, λ, options; blocks = collect(1:length(st.curvature)),
                          rng = default_rng(), gaptol = _row_gaptol(T, options, false),
                          max_iter = options.max_iter) where {T}
  θ, C, K = st.θ, length(st.θ), st.data.K
  if options.relative_tol && iszero(st.response_scale) && max_iter > 0
    foreach(A -> fill!(A, zero(T)), θ)
    foreach(r -> fill!(r, zero(T)), st.residuals)
    return (; passes = 1, converged = true)
  end
  tol = options.opt_tol * (options.relative_tol ? max(st.response_scale, floatmin(T)) : one(T))
  g, v, scratch = zeros(T, K), zeros(T, K), zeros(T, K)
  full = true
  for pass in 1:max_iter
    current = full ? copy(blocks) : [b for b in blocks if _transfer_active(st, b)]
    options.order === :random && shuffle!(rng, current)
    change = zero(T)
    for b in current
      c, j = _transfer_block(b, C)
      H = st.curvature[j, c + 1]
      iszero(H) && continue
      _transfer_gradient!(g, st, c, j)
      U = θ[c + 1]
      for k in 1:K
        v[k] = U[j, k] - g[k] / H
      end
      _row_prox!(v, st.penalties[c + 1], j, λ / H, scratch)
      for k in 1:K
        delta = v[k] - U[j, k]
        iszero(delta) && continue
        h = options.relative_tol ? abs(delta) * sqrt(st.coordinate_curvature[j, k, c + 1]) : abs(delta)
        change = max(change, h)
        U[j, k] = v[k]
        _transfer_residuals!(st, c, j, k, delta)
      end
    end
    if change <= tol
      if full
        gap = gaptol > 0 ? _transfer_gap(st, λ, blocks) : nothing
        (gap === nothing || gap <= gaptol) && return (; passes = pass, converged = true)
      end
      full = true
    else
      full = false
    end
  end
  (; passes = max_iter, converged = false)
end

# Continuation from λmax without warm starts, then the final fit.
function _transfer_fit!(st::_TransferState{T}, λ::T, options; rng) where {T}
  passes = 0
  continuation = !options.warm_start && options.num_steps > 0
  if continuation
    λmax = maximum(_transfer_duals(st); init = zero(T))
    if 0 < λ < λmax
      for k in 0:options.num_steps-1
        level = exp(log(λmax) + k * (log(λ) - log(λmax)) / options.num_steps)
        passes += _transfer_solve!(st, level, options; rng, gaptol = 0).passes
      end
    end
  end
  r = _transfer_solve!(st, λ, options; rng, gaptol = _row_gaptol(T, options, continuation))
  (; passes = passes + r.passes, converged = r.converged)
end

_transfer_intercepts(d::_TransferData{T}, l, B) where {T} =
  d.intercept ? T[d.ybar[l * d.K + k] - dot(view(d.xbar, :, l * d.K + k), view(B, :, k)) for k in 1:d.K] :
    zeros(T, d.K)

function _transfer_fused(d::_TransferData{T}, B) where {T}
  W = zeros(T, size(B[1]))
  for l in eachindex(B)
    W .+= (T(d.nobs[l]) / d.N) .* B[l]
  end
  W
end

"""
    TransferMultitaskSolution

The joint-sparse transfer fit of [`transfer_multitask`](@ref), corrected by
[`debias`](@ref) when `correction` is not `nothing`. Domain `0` is the target.

- `W`: the reported `p × K` target slopes, the fused `Ŵᶠ = ∑ₗ (nₗ/N) B̂ˡ` or the
  corrected `Ŵᴰ = Ŵᶠ + D̂`; `intercept`: the target prediction intercepts
  `ȳ⁰ - x̄⁰'W`, zero without `hasintercept`.
- `fused`: `Ŵᶠ`; `correction`: `D̂`, or `nothing`.
- `B`: the domain blocks `B̂⁰, …, B̂ᴸ`; `contrasts`: `B̂ˡ - B̂⁰` for the sources;
  `domain_intercepts`: `ȳˡ - x̄ˡ'B̂ˡ` per domain.
- `λ₀`, `λ̃` (`NaN` without a correction), `contrast_weights` (the values
  `a₁, …, a_L` used), `nobs` (the target size `n₀`), `domain_nobs`
  (`n₀, …, n_L`) and `solver`.
- The fused stage: `objective` (the objective of [`transfer_multitask`](@ref) at
  the returned blocks), `gap` (the global relative duality gap there, `NaN` when
  unavailable), `iterations` and `converged`. With `solver = :admm`,
  `primal_residual` and `dual_residual` are the largest constraint residuals of
  the last iteration and `penalties` the final ADMM penalties
  `ρ₀₀, …, ρ₀ᴸ, ρ₁, …, ρ_L`; they are `NaN` and empty for coordinate descent.
- The correction stage: `correction_objective`, `correction_gap`,
  `correction_iterations` and `correction_converged`, the same for the
  correction problem; `NaN`, `NaN`, `0` and `false` without a correction.

`coef`, `predict` and `nobs` apply to it.
"""
struct TransferMultitaskSolution{T<:AbstractFloat}
  W::Matrix{T}
  intercept::Vector{T}
  fused::Matrix{T}
  correction::Union{Nothing, Matrix{T}}
  B::Vector{Matrix{T}}
  contrasts::Vector{Matrix{T}}
  domain_intercepts::Vector{Vector{T}}
  λ₀::T
  λ̃::T
  contrast_weights::Vector{T}
  nobs::Int
  domain_nobs::Vector{Int}
  hasintercept::Bool
  solver::Symbol
  objective::T
  gap::T
  iterations::Int
  converged::Bool
  primal_residual::T
  dual_residual::T
  penalties::Vector{T}
  correction_objective::T
  correction_gap::T
  correction_iterations::Int
  correction_converged::Bool
end

# The fused solution at θ, with its objective and global gap recomputed from
# residuals at the returned blocks and intercepts.
function _transfer_solution(d::_TransferData{T}, a::Vector{T}, λ::T, θ, iterations, done, gaptol, solver;
                            primal_residual = T(NaN), dual_residual = T(NaN), penalties = T[]) where {T}
  K, L = d.K, length(d.nobs) - 1
  B = [l == 0 ? copy(θ[1]) : θ[1] .+ θ[l + 1] for l in 0:L]
  β0 = [_transfer_intercepts(d, l, B[l + 1]) for l in 0:L]
  st = _TransferState(d, a)
  foreach(copyto!, st.θ, θ)
  for l in 0:L, k in 1:K
    t = l * K + k
    st.residuals[t] .= d.y[t] .- β0[l + 1][k] .- Vector(d.X[t] * B[l + 1][:, k])
  end
  blocks = 1:length(st.curvature)
  objective = sum(r -> sum(abs2, r), st.residuals) / (2 * T(d.N * K)) +
    λ * sum(blocks; init = zero(T)) do b
      c, j = _transfer_block(b, L + 1)
      _row_value(view(st.θ[c + 1], j, :), st.penalties[c + 1], j)
    end
  gap = something(_transfer_gap(st, λ, blocks), T(NaN))
  converged = done && (isnan(gap) || gaptol <= 0 || gap <= gaptol)
  W = _transfer_fused(d, B)
  TransferMultitaskSolution{T}(W, _transfer_intercepts(d, 0, W), copy(W), nothing, B, [copy(θ[l + 1]) for l in 1:L], β0,
                               λ, T(NaN), a, d.nobs[1], d.nobs, d.intercept, solver, objective, gap, iterations,
                               converged, primal_residual, dual_residual, penalties, T(NaN), T(NaN), 0, false)
end

######################################################################
#   ADMM
######################################################################

const _TRANSFER_ADMM = (; max_iter = 10_000, abstol = 1e-8, reltol = 1e-6, rho = 1.0, rho_max = 1e4,
                        balance = 10.0, increase = 2.0, decrease = 1.5, adapt = 1_000)

function _admm_settings(::Type{T}, admm) where {T}
  admm isa NamedTuple || throw(ArgumentError("admm must be a named tuple of settings."))
  for k in keys(admm)
    haskey(_TRANSFER_ADMM, k) || throw(ArgumentError("unsupported ADMM setting $k."))
  end
  s = merge(_TRANSFER_ADMM, admm)
  s.max_iter isa Integer && s.max_iter >= 0 && s.adapt isa Integer && s.adapt >= 0 ||
    throw(ArgumentError("the ADMM max_iter and adapt must be nonnegative integers."))
  all(x -> x isa Real && isfinite(x) && x >= 0, (s.abstol, s.reltol)) ||
    throw(ArgumentError("the ADMM tolerances must be finite and nonnegative."))
  all(x -> x isa Real && isfinite(x), (s.rho, s.rho_max, s.balance, s.increase, s.decrease)) &&
    0 < s.rho <= s.rho_max && s.balance > 1 && s.increase > 1 && s.decrease > 1 ||
    throw(ArgumentError("the ADMM penalties need 0 < rho ≤ rho_max < ∞, and balance, increase and decrease above one."))
  (; max_iter = Int(s.max_iter), abstol = T(s.abstol), reltol = T(s.reltol), rho = T(s.rho), rho_max = T(s.rho_max),
   balance = T(s.balance), increase = T(s.increase), decrease = T(s.decrease), adapt = Int(s.adapt))
end

# The least squares ‖Ỹ - X̃B‖²/2 of domain ℓ on its centered data: one matrix
# loss for a shared design, a loss per response otherwise. The proximal
# operator caches a thin SVD, so every penalty value reuses it.
function _admm_losses(d::_TransferData{T}, l) where {T}
  ts = l * d.K .+ (1:d.K)
  centered(t) = Matrix{T}(d.X[t]) .- transpose(view(d.xbar, :, t))
  d.shared[l + 1] && return LeastSquaresLoss(reduce(hcat, [d.y[t] .- d.ybar[t] for t in ts]), centered(first(ts)))
  [LeastSquaresLoss(d.y[t] .- d.ybar[t], centered(t)) for t in ts]
end

_admm_prox!(B, f::LeastSquaresLoss, v, γ) = prox!(B, f, v, γ)
function _admm_prox!(B, f::AbstractVector, v, γ)
  for k in eachindex(f)
    prox!(view(B, :, k), f[k], v[:, k], γ)
  end
  B
end

_frobenius_difference(A, B) = sqrt(sum(i -> abs2(A[i] - B[i]), eachindex(A, B)))

# Two-block ADMM of Appendix A: B = (B⁰, …, Bᴸ) and the copies Γ⁰ˡ = B⁰
# (ℓ = 0, …, L) and Γˡ = Bˡ (ℓ ≥ 1), with a scaled dual and a penalty per
# constraint. Constraint c is Γ⁰⁽ᶜ⁻¹⁾ for c ≤ L + 1 and Γᶜ⁻ᴸ⁻¹ after.
function _transfer_admm(d::_TransferData{T}, a::Vector{T}, λ::T, s) where {T}
  p, K, L = size(d.xbar, 1), d.K, length(d.nobs) - 1
  NK, m = T(d.N * K), 2L + 1
  f = [_admm_losses(d, l) for l in 0:L]
  B = [zeros(T, p, K) for _ in 0:L]
  Γ, U = [zeros(T, p, K) for _ in 1:m], [zeros(T, p, K) for _ in 1:m]
  old = [zeros(T, p, K) for _ in 1:m]
  ρ = fill(s.rho, m)
  primal, dual = fill(T(NaN), m), fill(T(NaN), m)
  v = zeros(T, p, K)
  side(c) = B[c <= L + 1 ? 1 : c - L]
  iterations, converged = 0, false
  for it in 1:s.max_iter
    iterations = it
    # B⁰ with all its copies, then each source block.
    total = sum(view(ρ, 1:L+1))
    fill!(v, zero(T))
    for c in 1:L+1
      v .+= (ρ[c] / total) .* (Γ[c] .- U[c])
    end
    _admm_prox!(B[1], f[1], v, inv(NK * total))
    for l in 1:L
      v .= Γ[L + 1 + l] .- U[L + 1 + l]
      _admm_prox!(B[l + 1], f[l + 1], v, inv(NK * ρ[L + 1 + l]))
    end
    # Row soft thresholds for Γ⁰⁰, and the exact pair update of eq. (37).
    foreach(copyto!, old, Γ)
    for j in 1:p
      nrm = sqrt(sum(k -> abs2(B[1][j, k] + U[1][j, k]), 1:K))
      h = nrm > λ / ρ[1] ? 1 - λ / (ρ[1] * nrm) : zero(T)
      for k in 1:K
        Γ[1][j, k] = h * (B[1][j, k] + U[1][j, k])
      end
      for l in 1:L
        c0, c1 = l + 1, L + 1 + l
        ρ0, ρ1 = ρ[c0], ρ[c1]
        μ = ρ0 * ρ1 / (ρ0 + ρ1)
        nrm = sqrt(sum(k -> abs2(B[l + 1][j, k] + U[c1][j, k] - B[1][j, k] - U[c0][j, k]), 1:K))
        h = nrm > λ * a[l] / μ ? 1 - λ * a[l] / (μ * nrm) : zero(T)
        for k in 1:K
          q0, q1 = B[1][j, k] + U[c0][j, k], B[l + 1][j, k] + U[c1][j, k]
          r = h * (q1 - q0)
          Γ[c0][j, k] = (ρ0 * q0 + ρ1 * q1 - ρ1 * r) / (ρ0 + ρ1)
          Γ[c1][j, k] = (ρ0 * q0 + ρ1 * q1 + ρ0 * r) / (ρ0 + ρ1)
        end
      end
    end
    # Scaled duals, residuals of eq. (38), and the stopping rule per constraint.
    done = true
    for c in 1:m
      U[c] .+= side(c) .- Γ[c]
      primal[c] = _frobenius_difference(side(c), Γ[c])
      dual[c] = ρ[c] * _frobenius_difference(old[c], Γ[c])
      atol = sqrt(T(p * K)) * s.abstol
      done &= primal[c] <= atol + s.reltol * max(norm(side(c)), norm(Γ[c])) &&
        dual[c] <= atol + s.reltol * ρ[c] * norm(U[c])
    end
    if done
      converged = true
      break
    end
    # Residual balancing for a finite horizon, rescaling the scaled duals.
    if it <= s.adapt
      for c in 1:m
        new = primal[c] > s.balance * dual[c] ? min(s.increase * ρ[c], s.rho_max) :
          dual[c] > s.balance * primal[c] ? ρ[c] / s.decrease : ρ[c]
        new == ρ[c] && continue
        U[c] .*= ρ[c] / new
        ρ[c] = new
      end
    end
  end
  θ = [l == 0 ? copy(Γ[1]) : Γ[L + 1 + l] .- Γ[l + 1] for l in 0:L]
  (; θ, iterations, converged, primal_residual = maximum(primal), dual_residual = maximum(dual), penalties = ρ)
end

######################################################################
#   The correction
######################################################################

# The target designs with the residuals Y⁰ - X⁰W as responses and weights
# 1/K, so that the loss is ‖·‖²_F/(2n₀K), on unstandardized columns.
function _correction_data(Xs, Y, W::Matrix{T}, intercept::Bool) where {T}
  K = size(W, 2)
  _MultitaskData(Xs, [Y[:, k] .- Xs[k] * W[:, k] for k in 1:K]; intercept, standardize_x = false,
                 task_weights = fill(one(T) / K, K), stacked = false)
end

_correction_state(dc::_MultitaskData{T}) where {T} =
  _RowLoss(dc), _RowComponent(zeros(T, size(dc.S)), _RowPenalty(:l2, zero(T), ones(T, size(dc.S, 1))))

# The correction at the solver's coefficients, certified at the returned point.
function _correction_result(dc::_MultitaskData{T}, loss, comp, λ̃, passes, done, gaptol) where {T}
  D = _row_coefficients(dc, comp)
  d0 = _task_intercepts(dc, D)
  R = _task_residuals(dc, D, d0)
  fresh = _RowLoss(dc, R, loss.norms, loss.coordinate_curvature, loss.curvature, loss.response_scale)
  Dm = Matrix(D)
  gap = something(_row_gap(fresh, _RowComponent(dc.S .* Dm, comp.penalty), λ̃, axes(Dm, 1)), T(NaN))
  objective = sum(dc.c[t] * sum(abs2, R[t]) / (2 * dc.nobs[t]) for t in eachindex(R)) +
    λ̃ * sum(j -> norm(view(Dm, j, :)), axes(Dm, 1); init = zero(T))
  (; D = Dm, intercept = d0, objective, gap, passes,
   converged = done && (isnan(gap) || gaptol <= 0 || gap <= gaptol))
end

function _correction_fit(Xs, Y, W::Matrix{T}, λ̃::T, options; intercept, rng) where {T}
  dc = _correction_data(Xs, Y, W, intercept)
  loss, comp = _correction_state(dc)
  r = _row_fit!(loss, comp, λ̃, options; rng)
  _correction_result(dc, loss, comp, λ̃, r.passes, r.converged,
                     _row_gaptol(T, options, !options.warm_start && options.num_steps > 0))
end

function _with_correction(fit::TransferMultitaskSolution{T}, r, λ̃::T) where {T}
  TransferMultitaskSolution{T}(fit.fused .+ r.D, r.intercept, fit.fused, r.D, fit.B, fit.contrasts,
                               fit.domain_intercepts, fit.λ₀, λ̃, fit.contrast_weights, fit.nobs, fit.domain_nobs,
                               fit.hasintercept, fit.solver, fit.objective, fit.gap, fit.iterations, fit.converged,
                               fit.primal_residual, fit.dual_residual, fit.penalties, r.objective, r.gap, r.passes,
                               r.converged)
end

######################################################################
#   Fits
######################################################################

"""
    transfer_multitask(target, sources, λ₀, options = CDOptions(); contrast_weights = :sqrt,
                       intercept = true, solver = :cd, admm = (;), rng = Random.default_rng())

TMTL(Fused) of Lim and Kolar (2026): joint-sparse transfer from source domains
`ℓ = 1, …, L` to a target domain `ℓ = 0`. Domain `ℓ` has `nₗ` observations of
the same `K` responses on the same `p` predictors,

    minimize  ∑ₗ ∑ₖ ‖yˡᵏ - β₀ˡᵏ - Xˡᵏ Bˡ[:, k]‖² / (2NK)
                + λ₀ ∑ⱼ ‖B⁰[j, :]‖₂ + λ₀ ∑ₗ≥₁ aₗ ∑ⱼ ‖Bˡ[j, :] - B⁰[j, :]‖₂

over the `p × K` domain blocks `B⁰, …, Bᴸ` and unpenalized intercepts `β₀ˡᵏ`
per domain and response, with `N = ∑ₗ nₗ` over all domains, the target
included. The target block is row sparse; the contrasts `Bˡ - B⁰` are row
sparse, the sources' blocks need not be. The reported estimate is the fused
`Ŵᶠ = ∑ₗ (nₗ/N) B̂ˡ`, not `B̂⁰`; target predictions use the intercepts
`ȳ⁰ - x̄⁰'Ŵᶠ`. [`debias`](@ref) adds the target correction (TMTL(Debiased)).

`target = (X⁰, Y⁰)` and `sources` is a vector of `(Xˡ, Yˡ)`, possibly empty.
`Yˡ` is `nₗ × K`; `Xˡ` is one `nₗ × p` design for all responses, or a vector of
`K` designs, one per response (task-specific designs, Appendix B). Sizes may
differ across domains; the responses of a domain share its `nₗ` rows. Designs
are not standardized, since the contrasts compare coefficients across domains
on the predictors' own scales. With `intercept`, columns and responses are
centered within each domain and response. `contrast_weights` is `:sqrt`
(`aₗ = 8√(nₗ/N)`), `:linear` (`aₗ = 8nₗ/N`) or a vector of `L` positive weights.
`λ₀` must be finite and nonnegative.

The default solver is coordinate descent in the reparametrization `θ⁰ = B⁰`,
`θˡ = Bˡ - B⁰` (Appendix C.1), whose penalty `λ₀‖θ⁰‖₂,₁ + ∑ₗ λ₀aₗ‖θˡ‖₂,₁`
separates over components and rows. A row of `θ⁰` enters every domain, and its
gradient and curvature are summed over them; a row of `θˡ` enters domain `ℓ`.
Each row step uses the largest coordinate curvature of the row, exact for a
shared design and a majorization otherwise. A coefficient whose column is
constant in one domain can be nonzero, since the others determine it; only
coefficients that enter no loss stay zero. `options` are those of
[`CDOptions`](@ref), as for the row solver of [`multitask_lasso`](@ref),
including continuation without warm starts; a positive `gaptol` requires the
global relative duality gap described in the guide.

`solver = :admm` uses the paper's two-block ADMM (Appendix A) with the exact
pair update of eq. (37), cached least-squares factorizations, residual
balancing for the first `adapt` iterations, and its constraint-wise primal
and dual residual stopping rule. `admm` overrides the settings
`(; max_iter = 10_000, abstol = 1e-8, reltol = 1e-6, rho = 1.0, rho_max = 1e4,
balance = 10.0, increase = 2.0, decrease = 1.5, adapt = 1_000)`; constraint
`c` stops when its primal residual is at most `√(pK) abstol + reltol ⋅ max(‖B‖, ‖Γ‖)`
and its dual residual at most `√(pK) abstol + reltol ⋅ ρ‖U‖`. ADMM reports
`θ⁰ = Γ⁰⁰` and the contrasts `Γˡ - Γ⁰ˡ`, so zero rows are exact. `options`
and `rng` are not used by ADMM.

Returns a [`TransferMultitaskSolution`](@ref). See [`TransferMultitaskPath`](@ref)
and [`CVTransferMultitaskPath`](@ref) for paths and tuning.
"""
function transfer_multitask(target, sources::AbstractVector, λ₀::Real, options::CDOptions = CDOptions();
                            contrast_weights = :sqrt, intercept::Bool = true, solver::Symbol = :cd,
                            admm = (;), rng::AbstractRNG = default_rng(), kwargs...)
  _no_keywords(kwargs)
  solver in (:cd, :admm) || throw(ArgumentError("solver must be :cd or :admm."))
  isfinite(λ₀) && λ₀ >= 0 || throw(ArgumentError("λ₀ must be finite and nonnegative."))
  d = _TransferData(target, sources; intercept)
  T = eltype(d.ybar)
  a = _contrast_weights(T, contrast_weights, d.nobs)
  λ = T(λ₀)
  if solver === :admm
    r = _transfer_admm(d, a, λ, _admm_settings(T, admm))
    return _transfer_solution(d, a, λ, r.θ, r.iterations, r.converged, 0, :admm;
                              primal_residual = r.primal_residual, dual_residual = r.dual_residual,
                              penalties = r.penalties)
  end
  st = _TransferState(d, a)
  r = _transfer_fit!(st, λ, options; rng)
  _transfer_solution(d, a, λ, st.θ, r.passes, r.converged,
                     _row_gaptol(T, options, !options.warm_start && options.num_steps > 0), :cd)
end

"""
    debias(fit::TransferMultitaskSolution, target, λ̃, options = CDOptions();
           rng = Random.default_rng())

TMTL(Debiased): the target correction of the fused fit `Ŵᶠ = fit.fused`,

    D̂ = argmin  ∑ₖ ‖y⁰ᵏ - d₀ᵏ - X⁰ᵏ(Ŵᶠ[:, k] + D[:, k])‖² / (2n₀K) + λ̃ ∑ⱼ ‖D[j, :]‖₂,

with unpenalized intercepts `d₀ᵏ` when the fit has them, and `Ŵᴰ = Ŵᶠ + D̂`.
It is a group lasso of the target residuals `Y⁰ - X⁰Ŵᶠ` on the unstandardized
target design, the same as `multi_lasso(X⁰, Y⁰ - X⁰Ŵᶠ, K λ̃; standardize_x = false)`
for a shared design. "Debiased" refers to the aggregation bias of `Ŵᶠ` in
point estimation, not to inference. The correction uses the same target data
as the fused fit, which `target = (X⁰, Y⁰)` must give again; `λ̃` must be finite
and nonnegative. A column constant in the target has a zero correction.

The row solver of [`multitask_lasso`](@ref) computes `D̂`, with `options` and its
relative duality gap. The fused diagnostics are kept; the correction's are
reported separately. A corrected fit is corrected again from `fit.fused`.
Returns a [`TransferMultitaskSolution`](@ref) with `W = Ŵᴰ` and target
intercepts `ȳ⁰ - x̄⁰'Ŵᴰ`.
"""
function debias(fit::TransferMultitaskSolution{T}, target, λ̃::Real, options::CDOptions = CDOptions();
                rng::AbstractRNG = default_rng(), kwargs...) where {T}
  _no_keywords(kwargs)
  isfinite(λ̃) && λ̃ >= 0 || throw(ArgumentError("λ̃ must be finite and nonnegative."))
  Xs, Y = _transfer_domain(target, "the target")
  p, K = size(fit.W)
  size(Y) == (fit.nobs, K) ||
    throw(DimensionMismatch("the target responses are $(size(Y, 1)) × $(size(Y, 2)), but the fit has $(fit.nobs) × $K"))
  all(X -> size(X) == (fit.nobs, p), Xs) ||
    throw(DimensionMismatch("the target designs must be $(fit.nobs) × $p"))
  eltype(Y) <: Real && all(X -> eltype(X) <: Real, Xs) ||
    throw(ArgumentError("the designs and the responses of the target must have real entries."))
  _check_finite(Y, "the responses of the target")
  foreach(X -> _check_finite(X isa SparseMatrixCSC ? nonzeros(X) : X, "a design of the target"), Xs)
  r = _correction_fit(Xs, Y, fit.fused, T(λ̃), options; intercept = fit.hasintercept, rng)
  _with_correction(fit, r, T(λ̃))
end

######################################################################
#   Paths and tuning
######################################################################

function _check_grid(levels, nλ, λminratio, name)
  if levels === nothing
    nλ >= 1 || throw(ArgumentError("n$name must be positive."))
    λminratio === nothing || 0 < λminratio <= 1 || throw(ArgumentError("$(name)minratio must be in (0, 1]."))
  end
  nothing
end

# A warm path with the per-component sequential strong rule: block (c, j) is
# kept when it is active or ‖g_cj‖ ≥ w_c(2λ - λ_prev), and every excluded block
# is checked after convergence, repairing violations.
function _transfer_path!(record, d::_TransferData{T}, a::Vector{T}, λpath, options; nλ = 100,
                         λminratio = nothing, rng = default_rng(), screening = true) where {T}
  st = _TransferState(d, a)
  nb, C = length(st.curvature), length(st.θ)
  p = size(d.xbar, 1)
  M = _transfer_duals(st)
  λmax = maximum(M; init = zero(T))
  λs = if λpath === nothing
    top = λmax * (1 + sqrt(eps(T)))
    ratio = T(something(λminratio, d.N < C * p ? 0.01 : 1e-4))
    nλ == 1 ? [top] : [top * ratio^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  else
    Vector{T}(λpath)
  end
  all(λ -> isfinite(λ) && λ >= 0, λs) || throw(ArgumentError("The penalty levels must be finite and nonnegative."))
  issorted(λs; rev = true) || throw(ArgumentError("The penalty levels must be decreasing."))
  gaptol = _row_gaptol(T, options, true)
  checks = Int[]
  previous = isempty(λs) ? zero(T) : first(λs)
  for (i, λ) in enumerate(λs)
    threshold = 2λ - previous
    candidates = !screening || i == 1 ? collect(1:nb) :
      [b for b in 1:nb if _transfer_active(st, b) || M[(b - 1) ÷ C + 1, (b - 1) % C + 1] >= threshold]
    passes, rounds, done = 0, 0, false
    while true
      r = _transfer_solve!(st, λ, options; blocks = candidates, rng, gaptol, max_iter = options.max_iter - passes)
      passes += r.passes
      M = _transfer_duals(st)
      done = r.converged
      (!screening || !done) && break
      rounds += 1
      included = falses(nb); included[candidates] .= true
      violators = [b for b in 1:nb if !included[b] && M[(b - 1) ÷ C + 1, (b - 1) % C + 1] > λ]
      isempty(violators) && break
      append!(candidates, violators)
    end
    record(st.θ, λ, passes, done)
    push!(checks, rounds)
    previous = λ
  end
  (; levels = λs, checks, λmax)
end

"""
    TransferMultitaskPath(target, sources, options = CDOptions(); nλ = 100, λminratio = nothing, kwargs...)
    TransferMultitaskPath(target, sources, λpath, options = CDOptions(); contrast_weights = :sqrt,
                          intercept = true, screening = true, rng = Random.default_rng())

Fits of [`transfer_multitask`](@ref) by coordinate descent along a decreasing
sequence of levels `λ₀`, each warm started from the previous one, with fixed
contrast weights. The automatic grid has `nλ` log-spaced levels from
`λmax ⋅ (1 + √eps)` down to `λminratio` times that value; `λminratio` defaults to
`0.01` when `N < (L + 1)p` and to `1e-4` otherwise.

`λmax = max(maxⱼ ‖∑ₗ Sˡⱼ‖₂, maxₗ≥₁,ⱼ ‖Sˡⱼ‖₂ / aₗ)`, with scores
`Sˡⱼₖ = ⟨x̃ˡᵏⱼ, ỹˡᵏ⟩/(NK)` at zero coefficients, is the smallest level at which
every block, and not only `Ŵᶠ`, is zero. With `screening`, each level first
solves the blocks kept by the sequential strong rule for each component
(`θ⁰` with weight one, `θˡ` with weight `aₗ`), then checks the optimality of
every excluded block row and repairs (`kkt_checks` counts the rounds).

`λpath`, `solutions` (a [`TransferMultitaskSolution`](@ref) per level),
`kkt_checks`, `λmax`, `contrast_weights`, `nobs` and `domain_nobs` are its
fields. `coef(path)` and `predict(path, X; domain)` give a result per level;
`coef(path, λ)` requires an exact path level.
"""
struct TransferMultitaskPath{T<:AbstractFloat}
  λpath::Vector{T}
  solutions::Vector{TransferMultitaskSolution{T}}
  kkt_checks::Vector{Int}
  λmax::T
  contrast_weights::Vector{T}
  nobs::Int
  domain_nobs::Vector{Int}
end

function _transfer_fitpath(d::_TransferData{T}, a::Vector{T}, λpath, options; nλ = 100, λminratio = nothing,
                           screening = true, rng = default_rng()) where {T}
  solutions = TransferMultitaskSolution{T}[]
  gaptol = _row_gaptol(T, options, true)
  out = _transfer_path!(d, a, λpath, options; nλ, λminratio, screening, rng) do θ, λ, passes, done
    push!(solutions, _transfer_solution(d, a, λ, θ, passes, done, gaptol, :cd))
  end
  TransferMultitaskPath{T}(out.levels, solutions, out.checks, out.λmax, a, d.nobs[1], d.nobs)
end

function _transfer_path(target, sources::AbstractVector, λpath, options::CDOptions; contrast_weights = :sqrt,
                        intercept::Bool = true, nλ::Integer = 100, λminratio::Union{Nothing, Real} = nothing,
                        screening::Bool = true, rng::AbstractRNG = default_rng(), kwargs...)
  _no_keywords(kwargs)
  _check_grid(λpath, nλ, λminratio, "λ")
  d = _TransferData(target, sources; intercept)
  T = eltype(d.ybar)
  _transfer_fitpath(d, _contrast_weights(T, contrast_weights, d.nobs), λpath, options; nλ, λminratio, screening, rng)
end

TransferMultitaskPath(target, sources::AbstractVector, options::CDOptions = CDOptions(); kwargs...) =
  _transfer_path(target, sources, nothing, options; kwargs...)
TransferMultitaskPath(target, sources::AbstractVector, λpath::AbstractVector{<:Real},
                      options::CDOptions = CDOptions(); kwargs...) =
  _transfer_path(target, sources, λpath, options; kwargs...)

"""
    CVTransferMultitaskPath(target, sources, options = CDOptions(); families = (:sqrt, :linear),
                            nfolds = 5, folds = nothing, λpath = nothing, nλ = 100, λminratio = nothing,
                            λ̃path = nothing, nλ̃ = 100, λ̃minratio = nothing, intercept = true,
                            screening = true, rng = Random.default_rng(), threads = true)

Tune TMTL(Fused) and then its correction by held-out target prediction, with
coordinate descent and the options and screening of [`TransferMultitaskPath`](@ref).
`families` are contrast-weight families (`:sqrt`, `:linear` or a weight
vector; one symbol is one family), and `λpath` gives levels common to all
families. Folds are over the target rows, the same for every response, family,
level and correction level; every source row stays in every training fit. Each
fold refits on its target training rows and all sources, at the full-data grid
of each family, recomputing sizes, `N`, the contrast weights of `:sqrt` and
`:linear`, and the centering from that training data.

The held-out score of a fold is the target sum of squared errors divided by its
held-out rows times `K`, using target intercepts from the training rows; `cvm`
averages the folds weighted by their sizes, and `cvse` is its standard error.
`(family, λ₀)` minimizes `cvm` among the candidates converged on the full data
and in every fold; ties go to the earlier family, then the larger level. The
correction level `λ̃` is then selected with the same folds: each fold corrects
its own fused fit at the selected `(family, λ₀)` on its training residuals, so
the full-data fused fit is never used inside a fold. The correction grid
`λ̃path` defaults to `nλ̃` log-spaced levels from the full-data correction's
`λ̃max ⋅ (1 + √eps)`, the smallest level giving `D̂ = 0`, down to `λ̃minratio`
times that value (default `0.01` when `n₀ < p`, else `1e-4`), so a zero
correction is a candidate. Ties go to the larger `λ̃`.

Grids come from the full data, and the folds are reused for the sequential
selection, so `cvm` and `cvm_correction` are not unbiased estimates of the
selected fit's error; independent test data are needed for that.

Fields: `paths` (the full-data path of each family), `families`, `cvm` and
`cvse` (families × levels), `selected` (`CartesianIndex(family, level)`),
`family`, `λ₀`, `λ̃path`, `cvm_correction`, `cvse_correction`, `λ̃`, `fused`
(the full-data fused fit at the selection), `solution` (that fit with the
full-data correction at `λ̃`) and `folds`. `coef(cv)` and `predict(cv, X)` use
`solution`. With `threads`, folds run in parallel with generators seeded from
`rng`, so the result does not depend on the number of threads.
"""
struct CVTransferMultitaskPath{T<:AbstractFloat}
  paths::Vector{TransferMultitaskPath{T}}
  families::Vector{Union{Symbol, Vector{T}}}
  cvm::Matrix{T}
  cvse::Matrix{T}
  selected::CartesianIndex{2}
  family::Union{Symbol, Vector{T}}
  λ₀::T
  λ̃path::Vector{T}
  cvm_correction::Vector{T}
  cvse_correction::Vector{T}
  λ̃::T
  fused::TransferMultitaskSolution{T}
  solution::TransferMultitaskSolution{T}
  folds::Vector{Int}
end

function _target_sse(Xs, Y, W, b)
  sum(k -> sum(abs2, Y[:, k] .- b[k] .- Xs[k] * W[:, k]), axes(W, 2))
end

# The weighted mean and standard error of fold errors e (NaN where missing).
function _transfer_cvstats(e::AbstractVector{T}, m) where {T}
  ok = findall(isfinite, e)
  isempty(ok) && return T(NaN), T(NaN)
  w = T.(m[ok])
  μ = sum(w .* e[ok]) / sum(w)
  μ, length(ok) > 1 ? sqrt(sum(w .* abs2.(e[ok] .- μ)) / sum(w) / (length(ok) - 1)) : T(NaN)
end

function CVTransferMultitaskPath(target, sources::AbstractVector, options::CDOptions = CDOptions();
                                 families = (:sqrt, :linear), nfolds::Integer = 5, folds = nothing,
                                 λpath = nothing, nλ::Integer = 100, λminratio::Union{Nothing, Real} = nothing,
                                 λ̃path = nothing, nλ̃::Integer = 100, λ̃minratio::Union{Nothing, Real} = nothing,
                                 intercept::Bool = true, screening::Bool = true, rng::AbstractRNG = default_rng(),
                                 threads::Bool = true, kwargs...)
  _no_keywords(kwargs)
  _check_grid(λpath, nλ, λminratio, "λ")
  _check_grid(λ̃path, nλ̃, λ̃minratio, "λ̃")
  domains = [_transfer_domain(target, "the target"); [_transfer_domain(s, _domain_name(l)) for (l, s) in enumerate(sources)]]
  d = _TransferData(domains; intercept)
  T, K = eltype(d.ybar), d.K
  families isa Symbol && (families = (families,))
  isempty(families) && throw(ArgumentError("families must not be empty."))
  weights = [_contrast_weights(T, f, d.nobs) for f in families]
  fams = Union{Symbol, Vector{T}}[f isa Symbol ? f : weights[i] for (i, f) in enumerate(families)]
  n₀, nf = d.nobs[1], length(fams)
  fl, nk = _foldlabels(rng, n₀, nfolds, folds)
  m = [count(==(k), fl) for k in 1:nk]
  for k in 1:nk
    n₀ - m[k] >= 1 + intercept || throw(ArgumentError("fold $k leaves $(n₀ - m[k]) target training observations."))
  end
  seeds = rand(rng, UInt64, nk + 1, nf + 1)
  paths = [_transfer_fitpath(d, weights[i], λpath, options; nλ, λminratio, screening, rng = Xoshiro(seeds[end, i]))
           for i in 1:nf]
  nl = maximum(P -> length(P.λpath), paths)
  sse = fill(T(NaN), nf, nk, nl)
  Wk = Array{Matrix{T}}(undef, nf, nk, nl)
  gaptol = _row_gaptol(T, options, true)
  holdout(k) = findall(!=(k), fl), findall(==(k), fl)
  # Fold closures use their own counters; i and j below are the selection.
  function fused!(k)
    train, test = holdout(k)
    dk = _TransferData([_transfer_rows(domains[1], train); domains[2:end]]; intercept)
    Xtest, Ytest = _transfer_rows(domains[1], test)
    for f in 1:nf
      ak = _contrast_weights(T, fams[f], dk.nobs)
      level = 0
      _transfer_path!(dk, ak, paths[f].λpath, options; screening, rng = Xoshiro(seeds[k, f])) do θ, λ, passes, done
        level += 1
        sol = _transfer_solution(dk, ak, λ, θ, passes, done, gaptol, :cd)
        sol.converged || return
        Wk[f, k, level] = sol.W
        sse[f, k, level] = _target_sse(Xtest, Ytest, sol.W, sol.intercept)
      end
    end
  end
  _runfolds(fused!, nk, threads)
  cvm, cvse = fill(T(NaN), nf, nl), fill(T(NaN), nf, nl)
  selected = nothing
  for i in 1:nf, j in eachindex(paths[i].λpath)
    e = [sse[i, k, j] / (m[k] * K) for k in 1:nk]
    cvm[i, j], cvse[i, j] = _transfer_cvstats(e, m)
    all(isfinite, e) && paths[i].solutions[j].converged || continue
    (selected === nothing || cvm[i, j] < cvm[selected]) && (selected = CartesianIndex(i, j))
  end
  selected === nothing &&
    throw(ArgumentError("no fused candidate converged on the full data and in every fold; check the fits."))
  i, j = Tuple(selected)
  fused = paths[i].solutions[j]
  # The correction path on the full data gives the grid; each fold corrects its own fused fit.
  dc = _correction_data(domains[1]..., fused.fused, intercept)
  loss, comp = _correction_state(dc)
  full = []
  out = _row_path!(loss, comp, λ̃path, options; nλ = nλ̃, λminratio = λ̃minratio,
                   rng = Xoshiro(seeds[end, end]), screening) do D, λ, passes, done
    push!(full, _correction_result(dc, loss, comp, λ, passes, done, gaptol))
  end
  λ̃s = out.levels
  sse2 = fill(T(NaN), nk, length(λ̃s))
  function correction!(k)
    train, test = holdout(k)
    Xtr, Ytr = _transfer_rows(domains[1], train)
    Xtest, Ytest = _transfer_rows(domains[1], test)
    dck = _correction_data(Xtr, Ytr, Wk[i, k, j], intercept)
    lk, ck = _correction_state(dck)
    level = 0
    _row_path!(lk, ck, λ̃s, options; rng = Xoshiro(seeds[k, end]), screening) do D, λ, passes, done
      level += 1
      res = _correction_result(dck, lk, ck, λ, passes, done, gaptol)
      res.converged || return
      sse2[k, level] = _target_sse(Xtest, Ytest, Wk[i, k, j] .+ res.D, res.intercept)
    end
  end
  _runfolds(correction!, nk, threads)
  cvm2, cvse2 = fill(T(NaN), length(λ̃s)), fill(T(NaN), length(λ̃s))
  chosen = 0
  for l in eachindex(λ̃s)
    e = [sse2[k, l] / (m[k] * K) for k in 1:nk]
    cvm2[l], cvse2[l] = _transfer_cvstats(e, m)
    all(isfinite, e) && full[l].converged || continue
    (chosen == 0 || cvm2[l] < cvm2[chosen]) && (chosen = l)
  end
  chosen == 0 && throw(ArgumentError("no correction level converged on the full data and in every fold; check the fits."))
  solution = _with_correction(fused, full[chosen], λ̃s[chosen])
  CVTransferMultitaskPath{T}(paths, fams, cvm, cvse, selected, fams[i], paths[i].λpath[j], λ̃s, cvm2, cvse2,
                             λ̃s[chosen], fused, solution, fl)
end

######################################################################
#   StatsAPI
######################################################################

function _transfer_predict(X::AbstractMatrix, B, b)
  _checkcolumns(X, size(B, 1))
  Matrix(X * B) .+ transpose(b)
end

function _transfer_predict(Xs::AbstractVector{<:AbstractMatrix}, B, b)
  length(Xs) == size(B, 2) ||
    throw(DimensionMismatch("there are $(length(Xs)) designs, but the model has $(size(B, 2)) responses."))
  foreach(X -> _checkcolumns(X, size(B, 1)), Xs)
  all(X -> size(X, 1) == size(Xs[1], 1), Xs) || throw(DimensionMismatch("the designs must have the same rows."))
  reduce(hcat, [Vector(Xs[k] * B[:, k]) .+ b[k] for k in eachindex(Xs)])
end

function _transfer_domain_fit(sol::TransferMultitaskSolution, domain)
  domain === nothing && return sol.W, sol.intercept
  0 <= domain < length(sol.B) || throw(ArgumentError("domain must be in 0:$(length(sol.B) - 1)."))
  sol.B[domain + 1], sol.domain_intercepts[domain + 1]
end

"""
    coef(sol::TransferMultitaskSolution; domain = nothing)
    predict(sol::TransferMultitaskSolution, X; domain = nothing)

The target coefficients `W` (fused or corrected), with the target intercepts
`ȳ⁰ - x̄⁰'W` in a first row when fitted, and target predictions, an `n × K`
matrix. `X` is one design or a vector of `K` designs with the same rows. With
`domain = ℓ`, both use that domain's block `B̂ˡ` and intercepts instead
(`domain = 0` gives `B̂⁰`, not `W`). `nobs` is the target size `n₀`. Paths give a
result per level, and CV uses its `solution`.
"""
function StatsAPI.coef(sol::TransferMultitaskSolution; domain::Union{Nothing, Integer} = nothing)
  B, b = _transfer_domain_fit(sol, domain)
  sol.hasintercept ? [transpose(b); B] : copy(B)
end
StatsAPI.predict(sol::TransferMultitaskSolution, X; domain::Union{Nothing, Integer} = nothing) =
  _transfer_predict(X, _transfer_domain_fit(sol, domain)...)
StatsAPI.nobs(sol::TransferMultitaskSolution) = sol.nobs

StatsAPI.coef(path::TransferMultitaskPath; kwargs...) = [coef(s; kwargs...) for s in path.solutions]
function StatsAPI.coef(path::TransferMultitaskPath{T}, λ::Real; kwargs...) where {T}
  k = findfirst(==(T(λ)), path.λpath)
  k === nothing && throw(ArgumentError("λ must be a fitted path level."))
  coef(path.solutions[k]; kwargs...)
end
StatsAPI.predict(path::TransferMultitaskPath, X; kwargs...) = [predict(s, X; kwargs...) for s in path.solutions]
StatsAPI.nobs(path::TransferMultitaskPath) = path.nobs

StatsAPI.coef(cv::CVTransferMultitaskPath; kwargs...) = coef(cv.solution; kwargs...)
StatsAPI.predict(cv::CVTransferMultitaskPath, X; kwargs...) = predict(cv.solution, X; kwargs...)
StatsAPI.nobs(cv::CVTransferMultitaskPath) = nobs(cv.solution)

StatsAPI.fit(::Type{TransferMultitaskSolution}, target, sources::AbstractVector, args...; kwargs...) =
  transfer_multitask(target, sources, args...; kwargs...)
StatsAPI.fit(::Type{TransferMultitaskPath}, target, sources::AbstractVector, args...; kwargs...) =
  TransferMultitaskPath(target, sources, args...; kwargs...)
StatsAPI.fit(::Type{CVTransferMultitaskPath}, target, sources::AbstractVector, args...; kwargs...) =
  CVTransferMultitaskPath(target, sources, args...; kwargs...)
