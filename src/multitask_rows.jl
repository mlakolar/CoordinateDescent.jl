######################################################################
#
#   Row coordinate descent on the tasks' own designs
#
######################################################################

function _check_row_penalty(row_norm::Symbol, alpha::Real)
  row_norm in (:l2, :linf) || throw(ArgumentError("row_norm must be :l2 or :linf."))
  0 <= alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1]."))
  row_norm === :linf && !iszero(alpha) && throw(ArgumentError("row_norm = :linf requires alpha = 0."))
  nothing
end

# The loss cache is shared by coefficient components on the same designs.
struct _RowLoss{T, D<:_MultitaskData{T}}
  data::D
  residuals::Vector{Vector{T}}
  norms::Matrix{T}
  coordinate_curvature::Matrix{T}
  curvature::Vector{T}
  response_scale::T
end

# One coefficient component U and its row penalty.
struct _RowPenalty{T}
  row_norm::Symbol
  alpha::T
  weights::Vector{T}
end

struct _RowComponent{T}
  U::Matrix{T}
  penalty::_RowPenalty{T}
end

function _RowLoss(d::_MultitaskData{T}) where {T}
  p, q = size(d.S)
  norms, a, curvature = zeros(T, p, q), zeros(T, p, q), zeros(T, p)
  for t in 1:q
    u = findfirst(s -> d.X[s] === d.X[t], 1:t-1)
    for j in 1:p
      norms[j, t] = u === nothing ? _row_norm2(d.X[t], j, d.xbar[j, t]) : norms[j, u]
      iszero(d.S[j, t]) && continue
      a[j, t] = d.c[t] * norms[j, t] / (d.nobs[t] * d.S[j, t]^2)
      curvature[j] = max(curvature[j], a[j, t])
    end
  end
  residuals = [d.y[t] .- d.ybar[t] for t in 1:q]
  response_scale = norm(T[sqrt(d.c[t] / d.nobs[t]) * norm(residuals[t]) for t in 1:q])
  _RowLoss(d, residuals, norms, a, curvature, response_scale)
end

# The squared norm of a centered column, without storing that column.
function _row_norm2(X::AbstractMatrix{T}, j, m) where {T}
  s = zero(T)
  @inbounds for i in axes(X, 1)
    s += abs2(X[i, j] - m)
  end
  s
end

function _row_norm2(X::SparseMatrixCSC{T}, j, m) where {T}
  s = (size(X, 1) - length(nzrange(X, j))) * m^2
  @inbounds for i in nzrange(X, j)
    s += abs2(nonzeros(X)[i] - m)
  end
  s
end

# The centered column's inner product with r, and r minus a times the column.
function _row_dot(X::AbstractMatrix{T}, r, j, m) where {T}
  s = zero(T)
  @inbounds for i in eachindex(r)
    s += (X[i, j] - m) * r[i]
  end
  s
end

function _row_dot(X::SparseMatrixCSC{T}, r, j, m) where {T}
  s = iszero(m) ? zero(T) : -m * sum(r)
  @inbounds for i in nzrange(X, j)
    s += nonzeros(X)[i] * r[rowvals(X)[i]]
  end
  s
end

function _row_residual!(r, X::AbstractMatrix, j, m, a)
  @inbounds for i in eachindex(r)
    r[i] -= a * (X[i, j] - m)
  end
end

function _row_residual!(r, X::SparseMatrixCSC, j, m, a)
  if !iszero(m)
    @inbounds for i in eachindex(r)
      r[i] += a * m
    end
  end
  @inbounds for i in nzrange(X, j)
    r[rowvals(X)[i]] -= a * nonzeros(X)[i]
  end
end

# The loss gradient over the fitted coordinates of row j of any component.
function _row_gradient!(g, loss::_RowLoss{T}, j) where {T}
  d = loss.data
  for t in eachindex(g)
    s = d.S[j, t]
    g[t] = iszero(s) ? zero(T) :
      -d.c[t] * _row_dot(d.X[t], loss.residuals[t], j, d.xbar[j, t]) / (d.nobs[t] * s)
  end
  g
end

function _row_gradients(loss::_RowLoss{T}) where {T}
  G = similar(loss.data.S)
  for j in axes(G, 1)
    _row_gradient!(view(G, j, :), loss, j)
  end
  G
end

# The proximal step of a row: clipping at the l1 projection threshold for
# linf, or entrywise soft thresholding followed by l2 shrinkage.
function _row_prox!(v::Vector{T}, pen::_RowPenalty{T}, j, step, scratch) where {T}
  if pen.row_norm === :linf
    radius = step * pen.weights[j]
    iszero(radius) && return v
    for t in eachindex(v)
      scratch[t] = abs(v[t])
    end
    if sum(scratch) <= radius
      return fill!(v, zero(T))
    end
    sort!(scratch; rev = true)
    s, threshold = zero(T), zero(T)
    for k in eachindex(scratch)
      s += scratch[k]
      threshold = (s - radius) / k
      (k == length(scratch) || threshold >= scratch[k + 1]) && break
    end
    for t in eachindex(v)
      v[t] = copysign(min(abs(v[t]), threshold), v[t])
    end
  else
    a, c = step * pen.alpha, step * (1 - pen.alpha) * pen.weights[j]
    s = zero(T)
    for t in eachindex(v)
      v[t] = copysign(max(abs(v[t]) - a, zero(T)), v[t])
      s += abs2(v[t])
    end
    h = iszero(s) ? zero(T) : max(1 - c / sqrt(s), zero(T))
    v .*= h
  end
  v
end

# The exact root of norm(soft(g, a λ)) = c λ, by sorted breakpoints.
function _row_root(g::AbstractVector{T}, a::T, c::T) where {T}
  u = sort!(abs.(g); rev = true)
  isempty(u) && return zero(T)
  iszero(first(u)) && return zero(T)
  iszero(a) && return iszero(c) ? T(Inf) : norm(u) / c
  iszero(c) && return first(u) / a
  s, s2, avg, m2 = zero(T), zero(T), zero(T), zero(T)
  for k in eachindex(u)
    s += u[k]; s2 += u[k]^2
    delta = u[k] - avg
    avg += delta / k
    m2 += delta * (u[k] - avg)
    disc = c^2 * s2 - a^2 * k * max(m2, zero(T))
    disc < 0 && continue
    λ = s2 / (a * s + sqrt(disc))
    (k == length(u) || a * λ >= u[k + 1]) && return λ
  end
  T(Inf)
end

# The dual norm of a row's gradient; Inf for a nonzero unpenalized row.
function _row_dual(g, pen::_RowPenalty{T}, j) where {T}
  w = pen.weights[j]
  if pen.row_norm === :linf
    s = sum(abs, g)
    return iszero(w) ? (iszero(s) ? zero(T) : T(Inf)) : s / w
  end
  _row_root(g, pen.alpha, (1 - pen.alpha) * w)
end

_row_value(u, pen::_RowPenalty, j) = pen.row_norm === :linf ? pen.weights[j] * norm(u, Inf) :
  pen.alpha * sum(abs, u) + (1 - pen.alpha) * pen.weights[j] * norm(u)

# The relative duality gap on the candidate rows. Excluded rows are fixed at
# zero here; paths check their optimality before accepting the solution.
function _row_gap(loss::_RowLoss{T}, comp::_RowComponent{T}, λ, candidates) where {T}
  pen, d = comp.penalty, loss.data
  iszero(λ) && return nothing
  any(j -> iszero(pen.weights[j]) && iszero(pen.alpha), candidates) && return nothing
  P = sum(d.c[t] * sum(abs2, loss.residuals[t]) / (2 * d.nobs[t]) for t in eachindex(d.y))
  P += λ * sum(j -> _row_value(view(comp.U, j, :), pen, j), candidates; init = zero(T))
  iszero(P) && return nothing
  g, scale = zeros(T, size(comp.U, 2)), one(T)
  for j in candidates
    _row_gradient!(g, loss, j)
    ρ = _row_dual(g, pen, j)
    ρ > λ && (scale = min(scale, λ / ρ))
  end
  D = zero(T)
  for t in eachindex(d.y)
    r = loss.residuals[t]
    yr = sum(i -> (d.y[t][i] - d.ybar[t]) * r[i], eachindex(r))
    D += d.c[t] / d.nobs[t] * (scale * yr - scale^2 * sum(abs2, r) / 2)
  end
  isfinite(P) && isfinite(D) || return T(Inf)
  gap = P - D
  gap >= -8 * eps(T) * max(abs(P), abs(D)) || return T(Inf)
  max(gap, 4 * eps(T) * abs(P)) / max(abs(P), floatmin(T))
end

_row_gaptol(::Type{T}, options, path) where {T} =
  something(options.gaptol, path ? max(ProximalSolvers.PATH_GAPTOL, 64 * eps(T)) : 0.0)

# Full passes alternate with passes on the active rows; a full pass confirms
# the change rule and, when requested and available, the gap rule.
function _row_solve!(loss::_RowLoss{T}, comp::_RowComponent{T}, λ, options;
                      candidates = collect(axes(comp.U, 1)), rng = default_rng(),
                      gaptol = _row_gaptol(T, options, false), max_iter = options.max_iter) where {T}
  d, U = loss.data, comp.U
  if options.relative_tol && iszero(loss.response_scale) && max_iter > 0
    fill!(U, zero(T))
    foreach(r -> fill!(r, zero(T)), loss.residuals)
    return (; passes = 1, converged = true)
  end
  tol = options.opt_tol * (options.relative_tol ? max(loss.response_scale, floatmin(T)) : one(T))
  q = size(U, 2)
  g, v, scratch = zeros(T, q), zeros(T, q), zeros(T, q)
  full = true
  for pass in 1:max_iter
    rows = full ? copy(candidates) : [j for j in candidates if any(!iszero, view(U, j, :))]
    options.order === :random && shuffle!(rng, rows)
    change = zero(T)
    for j in rows
      L = loss.curvature[j]
      iszero(L) && continue
      _row_gradient!(g, loss, j)
      for t in 1:q
        v[t] = U[j, t] - g[t] / L
      end
      _row_prox!(v, comp.penalty, j, λ / L, scratch)
      for t in 1:q
        iszero(d.S[j, t]) && continue
        delta = v[t] - U[j, t]
        h = options.relative_tol ? abs(delta) * sqrt(loss.coordinate_curvature[j, t]) : abs(delta)
        change = max(change, h)
        iszero(delta) && continue
        U[j, t] = v[t]
        _row_residual!(loss.residuals[t], d.X[t], j, d.xbar[j, t], delta / d.S[j, t])
      end
    end
    if change <= tol
      if full
        gap = gaptol > 0 ? _row_gap(loss, comp, λ, candidates) : nothing
        (gap === nothing || gap <= gaptol) && return (; passes = pass, converged = true)
      end
      full = true
    else
      full = false
    end
  end
  (; passes = max_iter, converged = false)
end

# Original-scale coefficients and intercepts from a row component.
function _row_coefficients(d::_MultitaskData{T}, comp::_RowComponent{T}) where {T}
  B = zeros(T, size(d.S))
  for i in eachindex(B)
    iszero(d.S[i]) || (B[i] = comp.U[i] / d.S[i])
  end
  sparse(B)
end

function _row_fit(d::_MultitaskData{T}, ω, λ::T, options; row_norm, alpha, rng) where {T}
  isfinite(λ) || throw(ArgumentError("λ must be finite."))
  loss = _RowLoss(d)
  comp = _RowComponent(zeros(T, size(d.S)), _RowPenalty(row_norm, T(alpha), ω))
  passes = 0
  continuation = !options.warm_start && options.num_steps > 0
  G = continuation ? _row_gradients(loss) : nothing
  λmax = continuation ? maximum(j -> _row_dual(view(G, j, :), comp.penalty, j),
                                axes(d.S, 1); init = zero(T)) : zero(T)
  if continuation && 0 < λ < λmax < Inf
    for k in 0:options.num_steps-1
      level = exp(log(λmax) + k * (log(λ) - log(λmax)) / options.num_steps)
      r = _row_solve!(loss, comp, level, options; rng, gaptol = 0)
      passes += r.passes
    end
  end
  r = _row_solve!(loss, comp, λ, options; rng, gaptol = _row_gaptol(T, options, continuation))
  B = _row_coefficients(d, comp)
  MultitaskLassoSolution{T}(B, _task_intercepts(d, B), λ, _activerows(B), passes + r.passes, r.converged,
                            d.intercept, d.N, d.nobs, d.c, ω, row_norm, T(alpha))
end

function _row_path(d::_MultitaskData{T}, ω, λpath, options; nλ = 100, λminratio = nothing,
                    row_norm = :l2, alpha = 0, rng = default_rng(), screening = true, max_rows = Inf) where {T}
  loss = _RowLoss(d)
  comp = _RowComponent(zeros(T, size(d.S)), _RowPenalty(row_norm, T(alpha), ω))
  G = _row_gradients(loss)
  p, q = size(G)
  λs = if λpath === nothing
    λmax = maximum(j -> _row_dual(view(G, j, :), comp.penalty, j), 1:p; init = zero(T))
    isfinite(λmax) || throw(ArgumentError("an automatic grid needs a finite λmax; give λpath for unpenalized rows."))
    top = λmax * (1 + sqrt(eps(T)))
    ratio = T(something(λminratio, d.N < p * q ? 0.01 : 1e-4))
    nλ == 1 ? [top] : [top * ratio^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  else
    Vector{T}(λpath)
  end
  all(λ -> isfinite(λ) && λ >= 0, λs) || throw(ArgumentError("The penalty levels must be finite and nonnegative."))
  issorted(λs; rev = true) || throw(ArgumentError("The penalty levels must be decreasing."))
  levels, Bpath, β0path = T[], SparseMatrixCSC{T, Int}[], Vector{T}[]
  iterations, converged, checks = Int[], Bool[], Int[]
  previous = isempty(λs) ? zero(T) : first(λs)
  for (k, λ) in enumerate(λs)
    threshold = 2 * λ - previous
    candidates = !screening || k == 1 ? collect(1:p) :
      [j for j in 1:p if any(!iszero, view(comp.U, j, :)) || _row_dual(view(G, j, :), comp.penalty, j) >= threshold]
    passes, rounds, done = 0, 0, false
    while true
      r = _row_solve!(loss, comp, λ, options; candidates, rng, gaptol = _row_gaptol(T, options, true),
                       max_iter = options.max_iter - passes)
      passes += r.passes
      G = _row_gradients(loss)
      done = r.converged
      (!screening || !done) && break
      rounds += 1
      included = falses(p); included[candidates] .= true
      violators = [j for j in 1:p if !included[j] && _row_dual(view(G, j, :), comp.penalty, j) > λ]
      isempty(violators) && break
      append!(candidates, violators)
    end
    B = _row_coefficients(d, comp)
    push!(levels, λ); push!(Bpath, B); push!(β0path, _task_intercepts(d, B))
    push!(iterations, passes); push!(converged, done); push!(checks, rounds)
    previous = λ
    length(_activerows(B)) > max_rows && break
  end
  MultitaskLassoPath{T}(levels, Bpath, β0path, _activerows.(Bpath), iterations, converged, checks, d.intercept,
                        d.N, d.nobs, d.c, ω, row_norm, T(alpha))
end
