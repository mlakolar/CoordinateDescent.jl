######################################################################
#
#   The dirty multitask model
#
######################################################################

function _check_dirty_norm(shared_norm::Symbol)
  shared_norm in (:linf, :l2) || throw(ArgumentError("shared_norm must be :linf or :l2."))
  nothing
end

function _check_dirty_ratio(ratio::Real)
  isfinite(ratio) && ratio > 0 || throw(ArgumentError("ratio must be positive and finite."))
  nothing
end

# Each row owns its SLOPE buffers, restricted to the fitted coordinates.
struct _DirtyRowPenalty{T}
  ratio::T
  shared_norm::Symbol
  weights::Vector{T}
  fitted::Vector{Vector{Int}}
  slopes::Vector{SLOPE{Vector{T}}}
end

function _DirtyRowPenalty(d::_MultitaskData{T}, ω, ratio, shared_norm) where {T}
  fitted = [findall(>(0), view(d.S, j, :)) for j in axes(d.S, 1)]
  slopes = [SLOPE(T[ω[j] * min(one(T), max(ratio - (i - 1), zero(T))) for i in eachindex(fitted[j])])
            for j in eachindex(fitted)]
  _DirtyRowPenalty(T(ratio), shared_norm, ω, fitted, slopes)
end

# The clipping threshold of the canonical split, on standardized coordinates.
function _dirty_threshold(v::AbstractVector{T}, r, shared_norm) where {T}
  x = sort!(abs.(v); rev = true)
  shared_norm === :linf && return x[ceil(Int, r)]
  for k in 0:length(x)-1
    r^2 > k || break
    τ = norm(view(x, k+1:length(x))) / sqrt(r^2 - k)
    lo, hi = x[k + 1], k == 0 ? T(Inf) : x[k]
    lo * (1 - 8eps(T)) <= τ <= hi * (1 + 8eps(T)) && return clamp(τ, lo, hi)
  end
  error("could not locate the dirty l2 split threshold")
end

# The most-shared interior split, with pure splits at the regime endpoints.
function _dirty_split(v::AbstractVector{T}, λb, λs, shared_norm) where {T}
  iszero(λb) && return copy(v), zeros(T, length(v))
  iszero(λs) && return zeros(T, length(v)), copy(v)
  r = λb / λs
  endpoint = shared_norm === :linf ? length(v) : sqrt(count(!iszero, v))
  r >= endpoint && return zeros(T, length(v)), copy(v)
  r <= 1 && return copy(v), zeros(T, length(v))
  τ = _dirty_threshold(v, r, shared_norm)
  a = clamp.(v, -τ, τ)
  a, v .- a
end

# The proximal step from projection onto the intersection of l2 and linf balls.
function _dirty_clipped_norm(v::AbstractVector{T}, a, d) where {T}
  s = zero(T)
  for x in v
    s = hypot(s, min(a * abs(x), d))
  end
  s
end

function _dirty_l2_prox!(v::AbstractVector{T}, c, d) where {T}
  (iszero(c) || iszero(d)) && return v
  clipped = _dirty_clipped_norm(v, one(T), d)
  a = one(T)
  if clipped > c
    lo, hi = zero(T), one(T)
    for _ in 1:80
      mid = (lo + hi) / 2
      (mid == lo || mid == hi) && break
      if _dirty_clipped_norm(v, mid, d) > c
        hi = mid
      else
        lo = mid
      end
    end
    a = lo
  end
  for t in eachindex(v)
    v[t] -= clamp(a * v[t], -d, d)
  end
  v
end

function _row_prox!(v::Vector{T}, pen::_DirtyRowPenalty{T}, j, step, scratch) where {T}
  (iszero(step) || iszero(pen.weights[j])) && return v
  u = view(v, pen.fitted[j])
  if pen.shared_norm === :linf
    prox!(u, pen.slopes[j], u, step)
  else
    _dirty_l2_prox!(u, step * pen.weights[j] * pen.ratio, step * pen.weights[j])
  end
  v
end

function _row_value(u, pen::_DirtyRowPenalty{T}, j) where {T}
  v = view(u, pen.fitted[j])
  pen.shared_norm === :linf && return pen.slopes[j](v)
  a, s = _dirty_split(v, pen.ratio, one(T), :l2)
  anorm = norm(a)
  pen.weights[j] * ((iszero(anorm) ? zero(T) : pen.ratio * anorm) + sum(abs, s))
end

function _row_dual(g, pen::_DirtyRowPenalty{T}, j) where {T}
  v = view(g, pen.fitted[j])
  top = maximum(abs, v; init = zero(T))
  iszero(top) && return zero(T)
  (iszero(pen.weights[j]) || iszero(pen.ratio)) && return T(Inf)
  dual = pen.shared_norm === :linf ? sum(abs, v) : norm(v)
  max(dual / pen.ratio, top) / pen.weights[j]
end

_row_gap_available(pen::_DirtyRowPenalty, candidates) =
  !iszero(pen.ratio) && all(j -> !iszero(pen.weights[j]), candidates)

"""
    DirtyMultitaskSolution

The dirty multitask fit of [`dirty_multitask`](@ref). `B`, `A` and `S` are sparse
`p × q` matrices on the data scale, with `B = A + S`. `intercept` holds the task
intercepts, zero without `hasintercept`. `λ_shared`, `λ_specific`, `shared_norm`,
`task_weights` and `penalty_factors` record the objective. `activerows` are the
nonzero rows of `B`; `sharedrows` are those of the canonical split `A`, not
identified shared features.

`objective` is the two-component objective. `gap` is the relative duality gap,
or `NaN` when unavailable. `kkt` is the largest normalized violation of dual
feasibility, the two Fenchel equalities, intercept stationarity, `B = A + S`
and zeros outside the fitted coordinates, recomputed at the returned point.
Each residual is divided by the maximum of one and the magnitudes of its terms.
`iterations` counts row passes; `converged` requires the change rule and any
requested, available gap target. `nobs = sum(task_nobs)` counts all tasks'
observations, unlike `multi_lasso`'s observation count. `dof` counts nonzero
entries of `B` plus intercepts; it is not effective degrees of freedom.
"""
struct DirtyMultitaskSolution{T<:AbstractFloat}
  B::SparseMatrixCSC{T, Int}
  A::SparseMatrixCSC{T, Int}
  S::SparseMatrixCSC{T, Int}
  intercept::Vector{T}
  λ_shared::T
  λ_specific::T
  shared_norm::Symbol
  activerows::Vector{Int}
  sharedrows::Vector{Int}
  objective::T
  gap::T
  kkt::T
  iterations::Int
  converged::Bool
  hasintercept::Bool
  nobs::Int
  task_nobs::Vector{Int}
  task_weights::Vector{T}
  penalty_factors::Vector{T}
end

# Components on the data scale, splitting only the fitted entries of each row.
function _dirty_components(d::_MultitaskData{T}, B, λb, λs, shared_norm) where {T}
  A, S = zeros(T, size(B)), zeros(T, size(B))
  for j in axes(B, 1)
    on = findall(>(0), view(d.S, j, :))
    v = T[d.S[j, t] * B[j, t] for t in on]
    a, s = _dirty_split(v, λb, λs, shared_norm)
    for (k, t) in enumerate(on)
      A[j, t] = a[k] / d.S[j, t]
      S[j, t] = s[k] / d.S[j, t]
    end
  end
  sparse(A), sparse(S)
end

# The two-component certificate at the returned coefficients and intercepts.
function _dirty_certificate(d::_MultitaskData{T}, B, A, S, β0, λb, λs, pen, R) where {T}
  err, objective = zero(T), zero(T)
  Z = zeros(T, size(B))
  for t in eachindex(R)
    objective += d.c[t] * sum(abs2, R[t]) / (2 * d.nobs[t])
    if d.intercept
      h = d.c[t] * mean(R[t])
      err = max(err, abs(h) / max(one(T), d.c[t] * norm(R[t]) / sqrt(d.nobs[t])))
    end
    for j in axes(B, 1)
      iszero(d.S[j, t]) && continue
      Z[j, t] = d.c[t] * _row_dot(d.X[t], R[t], j, d.xbar[j, t]) / (d.nobs[t] * d.S[j, t])
    end
  end
  for j in axes(B, 1)
    on = pen.fitted[j]
    z = view(Z, j, on)
    a, s = d.S[j, on] .* Vector(A[j, on]), d.S[j, on] .* Vector(S[j, on])
    anorm = pen.shared_norm === :linf ? maximum(abs, a; init = zero(T)) : norm(a)
    b, c = pen.weights[j] * λb, pen.weights[j] * λs
    dual = pen.shared_norm === :linf ? sum(abs, z) : norm(z)
    top = maximum(abs, z; init = zero(T))
    za, zs, pa, ps = dot(z, a), dot(z, s), b * anorm, c * sum(abs, s)
    err = max(err, (dual - b) / max(one(T), dual, b), (top - c) / max(one(T), top, c),
              abs(za - pa) / max(one(T), abs(za), pa), abs(zs - ps) / max(one(T), abs(zs), ps))
    objective += pa + ps
    for t in axes(B, 2)
      scale = max(one(T), abs(B[j, t]), abs(A[j, t]), abs(S[j, t]))
      err = max(err, abs(B[j, t] - A[j, t] - S[j, t]) / scale)
      iszero(d.S[j, t]) && (err = max(err, abs(B[j, t]), abs(A[j, t]), abs(S[j, t])))
    end
  end
  (; objective, kkt = err)
end

function _dirty_solution(loss::_RowLoss{T}, comp, B, λb, λs, passes, done, gaptol) where {T}
  d, pen = loss.data, comp.penalty
  β0 = _task_intercepts(d, B)
  A, S = _dirty_components(d, B, λb, λs, pen.shared_norm)
  R = _task_residuals(d, B, β0)
  cert = _dirty_certificate(d, B, A, S, β0, λb, λs, pen, R)
  fresh = _RowLoss(d, R, loss.norms, loss.coordinate_curvature, loss.curvature, loss.response_scale)
  returned = _RowComponent(d.S .* Matrix(B), pen)
  gap = something(_row_gap(fresh, returned, λs, axes(B, 1)), T(NaN))
  converged = done && (isnan(gap) || gaptol <= 0 || gap <= gaptol)
  DirtyMultitaskSolution{T}(B, A, S, β0, λb, λs, pen.shared_norm, _activerows(B), _activerows(A),
                           cert.objective, gap, cert.kkt, passes, converged, d.intercept, d.N, d.nobs, d.c,
                           pen.weights)
end

"""
    dirty_multitask(Xs, ys, λ_shared, λ_specific, options = CDOptions(); shared_norm = :linf,
                    intercept = true, standardize_x = true, task_weights = :equal,
                    penalty_factors = nothing, rng = Random.default_rng())
    dirty_multitask(X, Y, λ_shared, λ_specific, options = CDOptions(); kwargs...)

The dirty model of Jalali et al. (2010), with total coefficients `B = A + S`,

    minimize  ∑ₜ cₜ ‖yₜ - β₀ₜ - Xₜ(aₜ + sₜ)‖² / (2nₜ)
                + ∑ⱼ ωⱼ (λ_shared ‖Ã[j, :]‖∞ + λ_specific ‖S̃[j, :]‖₁).

Here `Ã[j, t] = scale[j, t] * A[j, t]`, and likewise for `S̃`, with scales,
task weights and unpenalized intercepts as in [`multitask_lasso`](@ref).
`Xs[t]` and `ys[t]` may have unequal task sizes; `X, Y` share one design, a
response per column. `shared_norm = :l2` replaces the shared norm by ℓ2, a
variant of the paper's model. Levels must be finite and nonnegative.
Penalty factors multiply both terms; a zero factor leaves a row unpenalized.

The solver minimizes over `B` using the infimal-convolution row penalty.
For ℓ∞ this is SLOPE with weights `ωⱼ min(λ_specific,
max(λ_shared - (i - 1)λ_specific, 0))` on the fitted coordinates. No stacked
design is built; sparse designs are centered on the fly. `options` controls
the row solver, including continuation when `warm_start = false`.

Components are split in standardized coordinates, then rescaled. A zero shared
level gives `A = B`; a zero specific level gives `S = B`; both zero give `A = B`.
At positive levels, the lasso endpoint is handled first (`A = 0`), then the
row-penalty endpoint (`S = 0`). Interior splits clip the standardized row at
its `ceil(λ_shared / λ_specific)`-th magnitude for ℓ∞ (the most-shared choice
at integer ratios), or at the positive root described in the guide for ℓ2.
The split is canonical and need not be identified by the data.

Returns [`DirtyMultitaskSolution`](@ref). See [`DirtyMultitaskPath`](@ref) and
[`CVDirtyMultitaskPath`](@ref) for warm starts and tuning.
"""
function dirty_multitask(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                          λ_shared::Real, λ_specific::Real, options::CDOptions = CDOptions();
                          shared_norm::Symbol = :linf, intercept::Bool = true, standardize_x::Bool = true,
                          task_weights = :equal, penalty_factors = nothing,
                          rng::AbstractRNG = default_rng(), kwargs...)
  _no_keywords(kwargs)
  _check_dirty_norm(shared_norm)
  all(λ -> isfinite(λ) && λ >= 0, (λ_shared, λ_specific)) ||
    throw(ArgumentError("the penalty levels must be finite and nonnegative."))
  d = _MultitaskData(Xs, ys; intercept, standardize_x, task_weights, stacked = false)
  T = eltype(d.S)
  λb, λs = T(λ_shared), T(λ_specific)
  all(isfinite, (λb, λs)) || throw(ArgumentError("the penalty levels must be finite on the data scale."))
  ω = _penalty_factors(T, penalty_factors, size(d.S, 1))
  pen = _DirtyRowPenalty(d, ω, iszero(λs) ? one(T) : λb / λs, shared_norm)
  loss, comp = _RowLoss(d), _RowComponent(zeros(T, size(d.S)), pen)
  r = _row_fit!(loss, comp, λs, options; rng)
  target = _row_gaptol(T, options, !options.warm_start && options.num_steps > 0)
  _dirty_solution(loss, comp, _row_coefficients(d, comp), λb, λs, r.passes, r.converged, target)
end

_dirty_tasks(X::AbstractMatrix, Y::AbstractMatrix) = (fill(X, size(Y, 2)), [Y[:, t] for t in axes(Y, 2)])

function dirty_multitask(X::AbstractMatrix, Y::AbstractMatrix, λ_shared::Real, λ_specific::Real,
                          options::CDOptions = CDOptions(); kwargs...)
  Xs, ys = _dirty_tasks(X, Y)
  dirty_multitask(Xs, ys, λ_shared, λ_specific, options; kwargs...)
end

"""
    DirtyMultitaskPath(Xs, ys[, λpath], options = CDOptions(); ratio, shared_norm = :linf,
                       nλ = 100, λminratio = nothing, screening = true, max_rows = Inf, kwargs...)
    DirtyMultitaskPath(X, Y[, λpath], options = CDOptions(); ratio, kwargs...)

A decreasing path of specific levels `λ`, with shared levels `ratio * λ`.
`ratio` must be positive and finite. Total coefficients warm start each level;
components are recomputed. The automatic grid has `nλ` log-spaced levels from
`λmax * (1 + √eps)` to `λminratio` times that value. `λminratio` defaults to
`0.01` if `sum(nₜ) < pq`, and `1e-4` otherwise.

For gradient `gⱼ` on fitted coordinates, `λmax = maxⱼ dⱼ(gⱼ(0))`, where
`dⱼ(g) = max(‖g‖₁ / ratio, ‖g‖∞) / ωⱼ` for ℓ∞, and replaces ℓ1 by ℓ2 for
the variant. A nonzero gradient on an unpenalized row requires explicit levels.
Screening uses this gauge in the sequential strong rule and checks all excluded
rows. `max_rows` stops after the first fit exceeding that many nonzero rows.

`λpath`, `Bpath`, `Apath`, `Spath`, `β0path`, `activerows`, `sharedrows`,
`objective`, `gap`, `kkt`, `iterations` and `converged` hold each level's results;
`kkt_checks` counts screening repair rounds. `ratio`, `shared_norm`, `intercept`,
`nobs`, `task_nobs`, `task_weights` and `penalty_factors` record the fit settings.
`coef(path)` gives a matrix per level; `coef(path, λ)` requires an exact path
level, preserving its decomposition and certificate. `predict(path, Xs)` gives
an `nₜ × L` matrix per task; `predict(path, X)` gives an `n × q` matrix per level.
"""
struct DirtyMultitaskPath{T<:AbstractFloat}
  λpath::Vector{T}
  Bpath::Vector{SparseMatrixCSC{T, Int}}
  Apath::Vector{SparseMatrixCSC{T, Int}}
  Spath::Vector{SparseMatrixCSC{T, Int}}
  β0path::Vector{Vector{T}}
  activerows::Vector{Vector{Int}}
  sharedrows::Vector{Vector{Int}}
  objective::Vector{T}
  gap::Vector{T}
  kkt::Vector{T}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
  ratio::T
  shared_norm::Symbol
  intercept::Bool
  nobs::Int
  task_nobs::Vector{Int}
  task_weights::Vector{T}
  penalty_factors::Vector{T}
end

function _dirty_path(Xs, ys, λpath, options; ratio, shared_norm::Symbol = :linf, nλ::Integer = 100,
                       λminratio::Union{Nothing, Real} = nothing, screening::Bool = true, max_rows = Inf,
                       intercept::Bool = true, standardize_x::Bool = true, task_weights = :equal,
                       penalty_factors = nothing, rng::AbstractRNG = default_rng(), kwargs...)
  _no_keywords(kwargs)
  _check_dirty_norm(shared_norm)
  _check_dirty_ratio(ratio)
  if λpath === nothing
    nλ >= 1 || throw(ArgumentError("nλ must be positive."))
    λminratio === nothing || 0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]."))
  end
  d = _MultitaskData(Xs, ys; intercept, standardize_x, task_weights, stacked = false)
  T = eltype(d.S)
  r = T(ratio)
  _check_dirty_ratio(r)
  ω = _penalty_factors(T, penalty_factors, size(d.S, 1))
  pen = _DirtyRowPenalty(d, ω, r, shared_norm)
  loss, comp = _RowLoss(d), _RowComponent(zeros(T, size(d.S)), pen)
  solutions = DirtyMultitaskSolution{T}[]
  function record(B, λ, passes, done)
    isfinite(r * λ) || throw(ArgumentError("the shared penalty levels must be finite."))
    push!(solutions, _dirty_solution(loss, comp, B, r * λ, λ, passes, done, _row_gaptol(T, options, true)))
  end
  out = _row_path!(record, loss, comp, λpath, options; nλ, λminratio, rng, screening, max_rows)
  DirtyMultitaskPath{T}(out.levels, [s.B for s in solutions], [s.A for s in solutions], [s.S for s in solutions],
                       [s.intercept for s in solutions], [s.activerows for s in solutions],
                       [s.sharedrows for s in solutions], [s.objective for s in solutions], [s.gap for s in solutions],
                       [s.kkt for s in solutions], out.iterations, [s.converged for s in solutions], out.checks,
                       r, shared_norm, intercept, d.N, d.nobs, d.c, ω)
end

DirtyMultitaskPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                   options::CDOptions = CDOptions(); kwargs...) = _dirty_path(Xs, ys, nothing, options; kwargs...)
DirtyMultitaskPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                   λpath::AbstractVector{<:Real}, options::CDOptions = CDOptions(); kwargs...) =
  _dirty_path(Xs, ys, λpath, options; kwargs...)

function DirtyMultitaskPath(X::AbstractMatrix, Y::AbstractMatrix, args...; kwargs...)
  Xs, ys = _dirty_tasks(X, Y)
  DirtyMultitaskPath(Xs, ys, args...; kwargs...)
end

# A path level as a complete two-component result.
function _dirty_solution(path::DirtyMultitaskPath{T}, k::Int) where {T}
  DirtyMultitaskSolution{T}(path.Bpath[k], path.Apath[k], path.Spath[k], path.β0path[k],
                           path.ratio * path.λpath[k], path.λpath[k], path.shared_norm, path.activerows[k],
                           path.sharedrows[k], path.objective[k], path.gap[k], path.kkt[k], path.iterations[k],
                           path.converged[k], path.intercept, path.nobs, path.task_nobs, path.task_weights,
                           path.penalty_factors)
end

"""
    CVDirtyMultitaskPath(Xs, ys, options = CDOptions(); ratios = nothing, shared_norm = :linf,
                         nfolds = 10, folds = nothing, rng = Random.default_rng(), threads = true,
                         nλ = 100, kwargs...)
    CVDirtyMultitaskPath(X, Y, options = CDOptions(); kwargs...)

Tune the ratio and specific level by held-out prediction error. The default
uses five geometric ratios on `[1, q]` for ℓ∞, or `[1, √q]` for ℓ2. All ratios
use the same per-task folds. The shared-design method holds out whole rows
across responses. Each training fit refits its preprocessing and task weights;
its grid is the corresponding full-data path's grid.

The score is that of [`CVMultitaskLassoPath`](@ref), weighted by full-data task
weights. Nonconverged fold levels are unavailable. Selection compares only
levels reached and converged in every fold and on the full data. Ties go to
the larger ratio, then the larger specific level; there is no 1se rule.

`paths` holds the full-data paths and `ratios` their ratios. `cvm` and `cvse`
are `length(ratios) × nλ` matrices, padded with `NaN` for shorter paths;
`cvm_task[t, i, j]` is task `t`'s held-out MSE at ratio `i`, level `j`.
Means use the folds that reach and converge at a level. `selected` is the
selected `CartesianIndex(ratio_index, level_index)`, `ratio` and `λmin` its
settings, `path` its full-data path, and `solution` its converged full-data fit.
`folds` stores the labels per task, common across responses for a shared design.
`coef(cv)` and `predict(cv, Xnew)` use `solution`.
"""
struct CVDirtyMultitaskPath{T<:AbstractFloat}
  paths::Vector{DirtyMultitaskPath{T}}
  ratios::Vector{T}
  cvm::Matrix{T}
  cvse::Matrix{T}
  cvm_task::Array{T, 3}
  selected::CartesianIndex{2}
  ratio::T
  λmin::T
  path::DirtyMultitaskPath{T}
  solution::DirtyMultitaskSolution{T}
  folds::Vector{Vector{Int}}
  task_weights::Vector{T}
end

function CVDirtyMultitaskPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                              options::CDOptions = CDOptions(); ratios = nothing, shared_norm::Symbol = :linf,
                              nfolds::Integer = 10, folds = nothing, rng::AbstractRNG = default_rng(),
                              threads::Bool = true, nλ::Integer = 100, intercept::Bool = true, kwargs...)
  _check_dirty_norm(shared_norm)
  n, _ = _task_sizes(Xs, ys, intercept)
  q = length(n)
  rs = ratios === nothing ? exp.(range(0, log(shared_norm === :linf ? q : sqrt(q)); length = 5)) : collect(ratios)
  isempty(rs) && throw(ArgumentError("ratios must not be empty."))
  foreach(_check_dirty_ratio, rs)
  nλ >= 1 || throw(ArgumentError("nλ must be positive."))
  fl = folds === nothing ? [_folds(rng, nt, nfolds) for nt in n] : _task_folds(folds, n)
  K, nr = maximum(maximum, fl), length(rs)
  mt = [count(==(k), fl[t]) for t in 1:q, k in 1:K]
  for t in 1:q, k in 1:K
    n[t] - mt[t, k] >= 1 + intercept ||
      throw(ArgumentError("fold $k leaves too few training observations in task $t."))
  end
  seeds = rand(rng, UInt64, K + 1, nr)
  paths = [DirtyMultitaskPath(Xs, ys, options; ratio = rs[i], shared_norm, nλ, intercept,
                              rng = Xoshiro(seeds[end, i]), kwargs...) for i in 1:nr]
  T = eltype(paths[1].λpath)
  cvm, cvse, cvm_task = fill(T(NaN), nr, nλ), fill(T(NaN), nr, nλ), fill(T(NaN), q, nr, nλ)
  eligible = falses(nr, nλ)
  c = paths[1].task_weights
  wk = T[sum(c[t] * mt[t, k] / n[t] for t in 1:q) for k in 1:K]
  for i in 1:nr
    λs = paths[i].λpath
    L = length(λs)
    sse = fill(T(NaN), q, K, L)
    function fold!(k)
      train = [findall(!=(k), fl[t]) for t in 1:q]
      test = [findall(==(k), fl[t]) for t in 1:q]
      fp = DirtyMultitaskPath([Xs[t][train[t], :] for t in 1:q], [ys[t][train[t]] for t in 1:q], λs, options;
                              ratio = rs[i], shared_norm, intercept, rng = Xoshiro(seeds[k, i]), kwargs...)
      Ŷ = predict(fp, [Xs[t][test[t], :] for t in 1:q])
      for j in eachindex(fp.λpath)
        fp.converged[j] || continue
        for t in 1:q
          sse[t, k, j] = sum(abs2, ys[t][test[t]] - Ŷ[t][:, j])
        end
      end
    end
    _runfolds(fold!, K, threads)
    for j in 1:L
      errs = T[sum(c[t] / n[t] * sse[t, k, j] for t in 1:q) / wk[k] for k in 1:K]
      ok = findall(isfinite, errs)
      if !isempty(ok)
        w = wk[ok]
        m = sum(w .* errs[ok]) / sum(w)
        cvm[i, j] = m
        length(ok) > 1 && (cvse[i, j] = sqrt(sum(w .* abs2.(errs[ok] .- m)) / sum(w) / (length(ok) - 1)))
      end
      eligible[i, j] = length(ok) == K && paths[i].converged[j]
      for t in 1:q
        kt = findall(k -> isfinite(sse[t, k, j]), 1:K)
        isempty(kt) || (cvm_task[t, i, j] = sum(sse[t, kt, j]) / sum(mt[t, kt]))
      end
    end
  end
  candidates = findall(eligible)
  isempty(candidates) && throw(ArgumentError("no candidate converged on the full data and every fold; check the fits."))
  selected = first(candidates)
  for index in candidates
    i, j = Tuple(index)
    a, b = Tuple(selected)
    if cvm[index] < cvm[selected] || cvm[index] == cvm[selected] &&
       (rs[i] > rs[a] || rs[i] == rs[a] && paths[i].λpath[j] > paths[a].λpath[b])
      selected = index
    end
  end
  i, j = Tuple(selected)
  path = paths[i]
  sol = _dirty_solution(path, j)
  CVDirtyMultitaskPath{T}(paths, T.(rs), cvm, cvse, cvm_task, selected, path.ratio, path.λpath[j], path, sol, fl, c)
end

function CVDirtyMultitaskPath(X::AbstractMatrix, Y::AbstractMatrix, options::CDOptions = CDOptions();
                              nfolds::Integer = 10, folds = nothing, rng::AbstractRNG = default_rng(),
                              intercept::Bool = true, kwargs...)
  Xs, ys = _dirty_tasks(X, Y)
  n, _ = _task_sizes(Xs, ys, intercept)
  fl, _ = _foldlabels(rng, n[1], nfolds, folds)
  CVDirtyMultitaskPath(Xs, ys, options; folds = fill(fl, length(ys)), rng, intercept, kwargs...)
end

"""
    coef(sol::DirtyMultitaskSolution)
    predict(sol::DirtyMultitaskSolution, Xs)
    predict(sol::DirtyMultitaskSolution, X)

Total coefficients `B`, with intercepts in the first row when fitted, and
predictions as a vector per task or an `n × q` matrix for a shared design.
`nobs` counts all task observations. `dof` counts nonzero entries of `B` plus
fitted intercepts, not effective degrees of freedom. Paths support `coef` and
`predict` at every level; CV uses its selected full-data solution.
"""
StatsAPI.coef(sol::DirtyMultitaskSolution) = sol.hasintercept ? _prependrow(sol.intercept, sol.B) : sol.B
StatsAPI.predict(sol::DirtyMultitaskSolution, Xs::AbstractVector{<:AbstractMatrix}) =
  _taskpredict(Xs, sol.B, sol.intercept)
function StatsAPI.predict(sol::DirtyMultitaskSolution, X::AbstractMatrix)
  _checkcolumns(X, size(sol.B, 1))
  Matrix(X * sol.B) .+ sol.intercept'
end
StatsAPI.nobs(sol::DirtyMultitaskSolution) = sol.nobs
StatsAPI.dof(sol::DirtyMultitaskSolution) = nnz(sol.B) + sol.hasintercept * size(sol.B, 2)

StatsAPI.coef(path::DirtyMultitaskPath) = [coef(_dirty_solution(path, k)) for k in eachindex(path.λpath)]
function StatsAPI.coef(path::DirtyMultitaskPath{T}, λ::Real) where {T}
  k = findfirst(==(T(λ)), path.λpath)
  k === nothing && throw(ArgumentError("λ must be a fitted path level."))
  coef(_dirty_solution(path, k))
end
function StatsAPI.predict(path::DirtyMultitaskPath{T}, Xs::AbstractVector{<:AbstractMatrix}) where {T}
  q = length(path.task_nobs)
  length(Xs) == q || throw(DimensionMismatch("there are $(length(Xs)) designs, but the path has $q tasks."))
  [_predictpath(T, [B[:, t] for B in path.Bpath], T[β0[t] for β0 in path.β0path], Xs[t]) for t in 1:q]
end
StatsAPI.predict(path::DirtyMultitaskPath, X::AbstractMatrix) =
  [predict(_dirty_solution(path, k), X) for k in eachindex(path.λpath)]
StatsAPI.nobs(path::DirtyMultitaskPath) = path.nobs
StatsAPI.dof(path::DirtyMultitaskPath) = nnz.(path.Bpath) .+ path.intercept * length(path.task_nobs)

StatsAPI.coef(cv::CVDirtyMultitaskPath) = coef(cv.solution)
StatsAPI.predict(cv::CVDirtyMultitaskPath, X) = predict(cv.solution, X)
StatsAPI.nobs(cv::CVDirtyMultitaskPath) = nobs(cv.solution)
StatsAPI.dof(cv::CVDirtyMultitaskPath) = dof(cv.solution)

StatsAPI.fit(::Type{DirtyMultitaskSolution}, X, y, args...; kwargs...) = dirty_multitask(X, y, args...; kwargs...)
StatsAPI.fit(::Type{DirtyMultitaskPath}, X, y, args...; kwargs...) = DirtyMultitaskPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{CVDirtyMultitaskPath}, X, y, args...; kwargs...) = CVDirtyMultitaskPath(X, y, args...; kwargs...)
