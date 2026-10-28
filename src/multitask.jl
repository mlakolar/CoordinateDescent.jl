######################################################################
#
#   The multitask lasso with task-specific designs
#
######################################################################

# The q tasks y_t = β₀ₜ + X_t b_t + e_t of a multitask fit, with designs X_t of
# nₜ rows and the same p columns, as one least-squares problem over vec(B),
# B = [b_1 … b_q], whose entry (t - 1)p + j is b_tj (the layout of multi_lasso):
#
#     ∑ₜ cₜ ‖y_t - β₀ₜ - X_t b_t‖² / (2nₜ) = ‖ỹ - Zθ‖² / (2N),   θ_tj = s_tj b_tj,
#
# at the intercepts β₀ₜ = ȳₜ - x̄ₜ'b_t, with N = ∑ₜ nₜ. Z is block diagonal and
# sparse: block t holds the columns of X_t, centered at their means x̄ₜ with an
# intercept, divided by their scales s_tj, and with its rows multiplied by
# √(N cₜ / nₜ), as y_t - ȳₜ is in ỹ. The scales are the standard deviations of
# the columns within each task with standardize_x (_loadings), and one
# otherwise; an entry with s_tj = 0 (a zero column of X_t, or with an intercept
# a constant one) has b_tj = 0 and no column in Z. The columns of Z are the
# other entries of vec(B), in order, and the groups of the row penalty are the
# rows of B among them. The scaling is applied when Z is built, so a model whose
# structure it does not preserve, such as a low-rank B, should build its blocks
# with standardize_x = false. With shared, tasks that share one dense design
# and have equal weights are left to the multi-response model of multi_lasso,
# and Z and ỹ are not built (nothing). With stacked = false they are not
# built for any design; row coordinate descent reads X directly.
struct _MultitaskData{T<:AbstractFloat, VX<:AbstractVector{<:AbstractMatrix{T}}}
  X::VX                                      # the designs, as coordinate descent reads them
  y::Vector{Vector{T}}                       # the responses
  nobs::Vector{Int}                          # nₜ
  N::Int                                     # ∑ₜ nₜ
  c::Vector{T}                               # the task weights cₜ
  intercept::Bool
  xbar::Matrix{T}                            # p × q, the column means within the tasks (zero without an intercept)
  ybar::Vector{T}                            # the means of the responses (zero without an intercept)
  S::Matrix{T}                               # p × q, the scales s_tj
  fitted::Vector{Int}                        # the entry of vec(B) of each column of Z
  groups::Vector{Vector{Int}}                # the columns of Z of each row of B with columns in Z
  rows::Vector{Int}                          # the row of B of each group
  Z::Union{Nothing, SparseMatrixCSC{T, Int}}
  ỹ::Union{Nothing, Vector{T}}
end

function _MultitaskData(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector}; intercept::Bool,
                        standardize_x::Bool, task_weights, shared::Bool = false, stacked::Bool = true)
  n, p = _task_sizes(Xs, ys, intercept)
  q = length(n)
  all(X -> eltype(X) <: Real, Xs) && all(y -> eltype(y) <: Real, ys) ||
    throw(ArgumentError("the designs and the responses must have real entries."))
  T = float(promote_type(mapreduce(eltype, promote_type, Xs), mapreduce(eltype, promote_type, ys)))
  # a design given for several tasks is converted once
  Xv = Vector{AbstractMatrix{T}}(undef, q)
  for t in 1:q
    u = findfirst(s -> Xs[s] === Xs[t], 1:t-1)
    Xv[t] = u === nothing ? _design(convert(AbstractMatrix{T}, Xs[t])) : Xv[u]
  end
  X = map(identity, Xv)
  y = [convert(Vector{T}, yt) for yt in ys]
  c = _task_weights(T, task_weights, n)
  xbar, ybar, S = zeros(T, p, q), zeros(T, q), zeros(T, p, q)
  for t in 1:q
    _check_finite(y[t], "the response of task $t")
    u = findfirst(s -> X[s] === X[t], 1:t-1)
    if u !== nothing
      xbar[:, t] = view(xbar, :, u); S[:, t] = view(S, :, u)
    else
      _check_finite(X[t] isa SparseMatrixCSC ? nonzeros(X[t]) : X[t], "the design of task $t")
      intercept && (xbar[:, t] = _column_means(X[t]))
      s = _loadings(X[t], intercept)
      S[:, t] = standardize_x ? s : T.(s .> 0)
    end
    intercept && (ybar[t] = mean(y[t]))
  end
  fitted = [(t - 1) * p + j for t in 1:q for j in 1:p if S[j, t] > 0]
  col = zeros(Int, p * q)
  col[fitted] = eachindex(fitted)
  groups, rows = Vector{Int}[], Int[]
  for j in 1:p
    G = [col[(t - 1) * p + j] for t in 1:q if col[(t - 1) * p + j] > 0]
    isempty(G) || (push!(groups, G); push!(rows, j))
  end
  N = sum(n)
  Z, ỹ = !stacked || shared && _isshared(X, c) ? (nothing, nothing) : _stacked_design(X, y, n, c, xbar, ybar, S, fitted)
  _MultitaskData{T, typeof(X)}(X, y, n, N, c, intercept, xbar, ybar, S, fitted, groups, rows, Z, ỹ)
end

# the numbers of observations nₜ of the tasks and their number of columns p,
# after checks of the sizes
function _task_sizes(Xs::AbstractVector, ys::AbstractVector, intercept::Bool)
  q = length(Xs)
  q >= 1 || throw(ArgumentError("there must be at least one task."))
  length(ys) == q || throw(DimensionMismatch("there are $q designs and $(length(ys)) responses"))
  p = size(Xs[1], 2)
  n = Vector{Int}(undef, q)
  for t in 1:q
    size(Xs[t], 2) == p ||
      throw(DimensionMismatch("the design of task $t has $(size(Xs[t], 2)) columns, but that of task 1 has $p"))
    n[t] = size(Xs[t], 1)
    length(ys[t]) == n[t] ||
      throw(DimensionMismatch("the design of task $t has $(n[t]) rows and its response has length $(length(ys[t]))"))
    n[t] >= 1 + intercept ||
      throw(ArgumentError("task $t has $(n[t]) observations; a fit$(intercept ? " with an intercept" : "") needs at least $(1 + intercept)."))
  end
  n, p
end

# the task weights cₜ: one for :equal, q nₜ / N for :observations, or the
# positive weights given
function _task_weights(::Type{T}, task_weights, n::Vector{Int}) where {T}
  q = length(n)
  task_weights === :equal && return ones(T, q)
  task_weights === :observations && return T[q * nt for nt in n] ./ T(sum(n))
  task_weights isa AbstractVector{<:Real} ||
    throw(ArgumentError("task_weights must be :equal, :observations or a vector of positive weights, but it is $(repr(task_weights))."))
  length(task_weights) == q || throw(DimensionMismatch("there are $q tasks and $(length(task_weights)) task weights"))
  all(c -> isfinite(c) && c > 0, task_weights) || throw(ArgumentError("the task weights must be positive and finite."))
  Vector{T}(task_weights)
end

# the penalty factors ωⱼ: one by default, or the factors given, finite,
# nonnegative and not all zero
function _penalty_factors(::Type{T}, penalty_factors, p::Int) where {T}
  penalty_factors === nothing && return ones(T, p)
  length(penalty_factors) == p ||
    throw(DimensionMismatch("penalty_factors has $(length(penalty_factors)) entries, but the designs have $p columns"))
  all(w -> isfinite(w) && w >= 0, penalty_factors) || throw(ArgumentError("penalty_factors must be finite and nonnegative."))
  any(>(0), penalty_factors) || throw(ArgumentError("at least one penalty factor must be positive."))
  Vector{T}(penalty_factors)
end

# whether the tasks share one dense design and have equal weights
_isshared(X, c) = !issparse(X[1]) && all(Xt -> Xt === X[1] || Xt == X[1], X) && all(==(c[1]), c)

# Z and ỹ of the tasks (see _MultitaskData), Z built column by column; a column
# of a sparse design that is not centered keeps its nonzeros only
function _stacked_design(X, y, n::Vector{Int}, c::Vector{T}, xbar::Matrix{T}, ybar::Vector{T}, S::Matrix{T},
                         fitted::Vector{Int}) where {T}
  p, q = size(S)
  N = sum(n)
  off = cumsum([0; n[1:end-1]])
  r = [sqrt(N * c[t] / n[t]) for t in 1:q]
  colptr = Vector{Int}(undef, length(fitted) + 1)
  colptr[1] = 1
  rowval, nzval = Int[], T[]
  buf = Vector{T}(undef, maximum(n))
  for (k, e) in enumerate(fitted)
    t = div(e - 1, p) + 1
    j = e - (t - 1) * p
    a, m, Xt = r[t] / S[j, t], xbar[j, t], X[t]
    if Xt isa SparseMatrixCSC && iszero(m)
      rv, nv = rowvals(Xt), nonzeros(Xt)
      for i in nzrange(Xt, j)
        iszero(nv[i]) && continue
        push!(rowval, off[t] + rv[i]); push!(nzval, a * nv[i])
      end
    else
      v = _densecolumn!(view(buf, 1:n[t]), Xt, j)
      for i in 1:n[t]
        h = v[i] - m
        iszero(h) && continue
        push!(rowval, off[t] + i); push!(nzval, a * h)
      end
    end
    colptr[k + 1] = length(rowval) + 1
  end
  Z = SparseMatrixCSC(N, length(fitted), colptr, rowval, nzval)
  Z, reduce(vcat, [r[t] .* (y[t] .- ybar[t]) for t in 1:q])
end

# column j of X in buf
_densecolumn!(buf, X::AbstractMatrix, j::Int) = copyto!(buf, view(X, :, j))

function _densecolumn!(buf, X::SparseMatrixCSC, j::Int)
  fill!(buf, zero(eltype(buf)))
  rv, nv = rowvals(X), nonzeros(X)
  for i in nzrange(X, j)
    buf[rv[i]] = nv[i]
  end
  buf
end

# The model of a multitask fit, a zero iterate, and the groups and their weights
# at level one. For a shared design, the multi-response model of multi_lasso,
# whose loss is that of the tasks divided by their common weight c, so that the
# weights are ωⱼ sⱼ / c; a row whose column is zero (sⱼ = 0) stays zero at any
# weight, and keeps ωⱼ / c. Otherwise least squares on Z, with the groups of the
# rows that have columns in Z.
function _multitask_problem(d::_MultitaskData{T}, ω::Vector{T}) where {T}
  if d.Z === nothing
    m, groups, x, _ = _multi_problem(d.X[1], reduce(hcat, d.y), d.intercept, false)
    s = view(d.S, :, 1)
    return m, x, groups, T[ω[j] * (iszero(s[j]) ? one(T) : s[j]) / d.c[1] for j in eachindex(ω)]
  end
  CDLeastSquaresLoss(d.ỹ, d.Z; intercept = false), AtomIterate(T, size(d.Z, 2), d.groups), d.groups, ω[d.rows]
end

# B (p × q, sparse) from the solution θ of the model of d: the coefficients of
# the columns of Z divided by their scales, or vec(B) for a shared design
function _task_coefficients(d::_MultitaskData{T}, θ::AbstractVector) where {T}
  p, q = size(d.S)
  d.Z === nothing && return sparse(reshape(Vector(θ), p, q))
  θv = Vector(θ)
  I, J, V = Int[], Int[], T[]
  for k in eachindex(θv)
    iszero(θv[k]) && continue
    e = d.fitted[k]
    t = div(e - 1, p) + 1
    j = e - (t - 1) * p
    push!(I, j); push!(J, t); push!(V, θv[k] / d.S[j, t])
  end
  sparse(I, J, V, p, q)
end

# the intercepts β₀ₜ = ȳₜ - x̄ₜ'b_t of the coefficients B, zero without an intercept
_task_intercepts(d::_MultitaskData{T}, B::SparseMatrixCSC) where {T} =
  d.intercept ? T[d.ybar[t] - dot(view(d.xbar, :, t), B[:, t]) for t in axes(B, 2)] : zeros(T, size(B, 2))

# the residuals y_t - β₀ₜ - X_t b_t of the tasks
_task_residuals(d::_MultitaskData, B::AbstractMatrix, β0::AbstractVector) =
  [d.y[t] .- β0[t] .- Vector(d.X[t] * B[:, t]) for t in eachindex(d.y)]

# the objective ∑ₜ cₜ ‖y_t - β₀ₜ - X_t b_t‖² / (2nₜ) + λ ∑ⱼ ωⱼ ‖(s_1j b_1j, …, s_qj b_qj)‖₂
function _task_objective(d::_MultitaskData{T}, B::AbstractMatrix, β0::AbstractVector, λ::Real,
                         ω::AbstractVector) where {T}
  R = _task_residuals(d, B, β0)
  loss = sum(d.c[t] * sum(abs2, R[t]) / (2 * d.nobs[t]) for t in eachindex(R))
  loss + λ * sum((ω[j] * norm(d.S[j, :] .* Vector(B[j, :])) for j in axes(B, 1)); init = zero(T))
end

# other keywords are rejected: the options of coordinate descent are passed as a CDOptions
_no_keywords(kwargs) =
  isempty(kwargs) || throw(ArgumentError("unsupported keyword argument $(first(keys(kwargs)))."))

"""
    MultitaskLassoSolution

Result returned by [`multitask_lasso`](@ref).

# Fields
- `B::SparseMatrixCSC`: the `p × q` matrix of coefficients, a column per task,
  on the scale of the data
- `intercept::Vector`: the unpenalized intercepts of the tasks, zero unless
  `intercept = true`
- `λ`: the penalty level
- `activerows::Vector{Int}`: the selected predictors, the rows of `B` with a
  nonzero coefficient; entries of a selected row can still be zero
- `iterations::Int`, `converged::Bool`: the passes of coordinate descent over
  the rows and whether it met its stopping rule
- `hasintercept::Bool`: whether the solver fitted intercepts
- `nobs::Int`, `task_nobs::Vector{Int}`: the number of observations of all the
  tasks, `N = ∑ₜ nₜ`, and those of each task, `nₜ`
- `task_weights::Vector`, `penalty_factors::Vector`: the task weights `cₜ` and
  the penalty factors `ωⱼ` of the fit
- `row_norm::Symbol`, `alpha`: the row norm and share of the entrywise penalty

[`coef`](@ref) (a `(p + 1) × q` matrix, the intercepts in the first row),
[`predict`](@ref) (a vector of fitted values per task), [`nobs`](@ref) and
[`dof`](@ref) apply to it.
"""
struct MultitaskLassoSolution{T}
  B::SparseMatrixCSC{T, Int}
  intercept::Vector{T}
  λ::T
  activerows::Vector{Int}
  iterations::Int
  converged::Bool
  hasintercept::Bool
  nobs::Int
  task_nobs::Vector{Int}
  task_weights::Vector{T}
  penalty_factors::Vector{T}
  row_norm::Symbol
  alpha::T
end

MultitaskLassoSolution{T}(B, intercept, λ, activerows, iterations, converged, hasintercept, nobs, task_nobs,
                         task_weights, penalty_factors) where {T} =
  MultitaskLassoSolution{T}(B, intercept, λ, activerows, iterations, converged, hasintercept, nobs, task_nobs,
                           task_weights, penalty_factors, :l2, zero(T))
MultitaskLassoSolution(B::SparseMatrixCSC{T, Int}, args::Vararg{Any, 10}) where {T} =
  MultitaskLassoSolution{T}(B, args...)

"""
    multitask_lasso(Xs, ys, λ, options = CDOptions(); intercept = true, standardize_x = true,
                    task_weights = :equal, penalty_factors = nothing, row_norm = :l2, alpha = 0,
                    rng = Random.default_rng())

The multitask lasso of the responses `ys[t]` on the designs `Xs[t]`, `t = 1, …, q`,
with the same `p` columns in every task (the same predictors, in the same order)
and `nₜ` rows, which may differ between the tasks,

    minimize  ∑ₜ cₜ ‖y_t - β₀ₜ - X_t b_t‖² / (2nₜ) + λ ∑ⱼ hⱼ((s₁ⱼ b₁ⱼ, …, s_qⱼ b_qⱼ))

over the intercepts `β₀ₜ` and the coefficients `B = [b_1 … b_q]`. The penalty
on the rows of `B` selects predictors across tasks; a selected row can have
zero entries. With `row_norm = :l2`,
`hⱼ(u) = alpha * ‖u‖₁ + (1 - alpha) * ωⱼ * ‖u‖₂`, the row ℓ1/ℓ2 penalty by
default (`alpha = 0`) and the multitask sparse group for `alpha > 0`.
With `row_norm = :linf`, `hⱼ(u) = ωⱼ * ‖u‖∞`, and `alpha` must be zero.
`alpha` must lie in `[0, 1]`.

- The task weights `cₜ` are given by `task_weights`: `:equal`, the default,
  `cₜ = 1`, so that the mean squared error of each task counts once;
  `:observations`, `cₜ = q nₜ / N` with `N = ∑ₜ nₜ`, so that each observation
  counts once (the loss is `q` times the pooled loss `∑ₜ ‖y_t - β₀ₜ - X_t b_t‖² / (2N)`);
  or a vector of positive weights. With equal sizes the first two are the same.
- With `standardize_x`, `sₜⱼ` is the standard deviation
  `sqrt(∑ᵢ (X_t[i, j] - x̄ₜⱼ)² / nₜ)` of column `j` within task `t`, centered at
  its mean `x̄ₜⱼ` with an intercept and at zero otherwise: the penalty is on the
  coefficients of the columns standardized within each task. Otherwise
  `sₜⱼ = 1`. Either way, a column that is zero in task `t`, or with an
  intercept constant there, has a zero coefficient in that task. The
  coefficients are on the scale of the data.
- The penalty factors `ωⱼ` are given by `penalty_factors`, finite, nonnegative
  and not all zero, one by default. They weight the row term only: a zero
  factor leaves the entrywise term when `alpha > 0`.
- The intercepts are not penalized, and they are zero unless `intercept` is true.

With unit penalty factors, one design `X` for all the tasks and equal task
weights `c`, this is [`multi_lasso`](@ref)`(X, [y_1 … y_q], λ / c)`, and with one
task it is [`lasso`](@ref)`(X_1, y_1, λ / c₁)`, both with the same `intercept`
and `standardize_x` and unit factors; these reductions hold for both new
penalties too.

For the default penalty, the tasks are stacked into one least-squares problem
over `vec(B)`, whose design has the centered and scaled designs of the tasks on its diagonal and is
stored as a sparse matrix; a sparse design stays sparse without an intercept,
and with an intercept its centered columns are dense. It is solved by block
coordinate descent on the rows of `B` (ProximalSolvers' group coordinate
descent), with `options`; with `options.warm_start` false the solution is
computed along `options.num_steps` levels from λmax. The random order of
coordinate descent is drawn from `rng`. When the tasks share one dense design
(the same matrix, or equal ones) and have equal weights, the model of
[`multi_lasso`](@ref) is used instead, on that design. Real data of any types
are converted to a common floating point type, and options are passed as
[`CDOptions`](@ref).

The new penalties use row coordinate descent on the tasks' own designs,
without stacking or copying a shared design. Sparse columns are centered on
the fly. Full passes alternate with active-row passes. With `relative_tol`,
the change rule divides the largest change of the standardized coefficients
by `max(1, max|U|)`; otherwise it uses absolute changes. Single fits require
a duality gap only with a positive `gaptol`; continuation uses the path target.
There is no gap at λ = 0, with completely unpenalized rows, or at zero objective.

Returns a [`MultitaskLassoSolution`](@ref). See also [`MultitaskLassoPath`](@ref)
and [`CVMultitaskLassoPath`](@ref).
"""
function multitask_lasso(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector}, λ::Real,
                         options::CDOptions = CDOptions(); intercept::Bool = true, standardize_x::Bool = true,
                         task_weights = :equal, penalty_factors = nothing, rng::AbstractRNG = default_rng(),
                         row_norm::Symbol = :l2, alpha::Real = 0, kwargs...)
  _no_keywords(kwargs)
  _multitask_fit(Xs, ys, λ, options; intercept, standardize_x, task_weights, penalty_factors, rng, row_norm, alpha)
end

# With the default penalty and shared = false, fit shared designs by stacking.
function _multitask_fit(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector}, λ::Real,
                        options::CDOptions = CDOptions(); intercept::Bool = true, standardize_x::Bool = true,
                        task_weights = :equal, penalty_factors = nothing, rng::AbstractRNG = default_rng(),
                        shared::Bool = true, row_norm::Symbol = :l2, alpha::Real = 0)
  _check_row_penalty(row_norm, alpha)
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  d = _MultitaskData(Xs, ys; intercept, standardize_x, task_weights, shared,
                     stacked = row_norm === :l2 && iszero(alpha))
  T = eltype(d.S)
  ω = _penalty_factors(T, penalty_factors, size(d.S, 1))
  row_norm === :l2 && iszero(alpha) || return _row_fit(d, ω, T(λ), options; row_norm, alpha, rng)
  m, x, groups, w = _multitask_problem(d, ω)
  r = _solve!(x, m, GroupNormL2(T(λ), groups, w), options; rng)
  B = _task_coefficients(d, x)
  MultitaskLassoSolution{T}(B, _task_intercepts(d, B), T(λ), _activerows(B), r.passes, r.converged, intercept, d.N,
                            d.nobs, d.c, ω)
end

"""
    MultitaskLassoPath(Xs, ys, options = CDOptions(); nλ = 100, λminratio, kwargs...)
    MultitaskLassoPath(Xs, ys, λpath, options = CDOptions(); intercept = true, standardize_x = true,
                       task_weights = :equal, penalty_factors = nothing, rng = Random.default_rng(),
                       screening = true, max_rows = Inf, row_norm = :l2, alpha = 0)

Solutions `β0path[k]` and `Bpath[k]` of the multitask lasso of
[`multitask_lasso`](@ref) (with the same keywords) along a decreasing sequence
`λpath`, each one warm started from the previous solution. The default uses
ProximalSolvers' `solve_path`; the new penalties use row coordinate descent
on the tasks' own designs.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when there are fewer observations than
coefficients, `∑ₜ nₜ < pq`, and to `1e-4` otherwise, the rule of
[`LassoPath`](@ref) for one task and of [`MultiLassoPath`](@ref) for a shared
design. The first value is `λmax` times `1 + √eps`. This grid needs positive
penalty factors for row ℓ1/ℓ2 and ℓ1/ℓ∞. Sparse group with `alpha > 0`
also allows zero factors, since the entries remain penalized. The grid needs
finite λmax; otherwise give `λpath`. For ℓ∞, λmax is the largest row gradient
ℓ1 norm divided by its factor; for sparse group it is the largest root of
`‖soft(gⱼ(0), alpha * λ)‖₂ = (1 - alpha) * ωⱼ * λ`.

With `screening`, each λ is solved first over the rows kept by the sequential
strong rule, and the optimality conditions of the others are checked
afterwards (`kkt_checks[k]` times at `λpath[k]`). The path stops early after the
first solution with more than `max_rows` nonzero rows; the returned `λpath`
then contains only the values that were solved for.

# Fields
- `λpath::Vector`: the levels solved for
- `Bpath::Vector{SparseMatrixCSC}`, `β0path::Vector{Vector}`: the `p × q`
  coefficients and the intercepts of the tasks at each level
- `activerows::Vector{Vector{Int}}`: the selected predictors at each level
- `iterations::Vector{Int}`, `converged::Vector{Bool}`, `kkt_checks::Vector{Int}`:
  the passes of coordinate descent at each level, whether it met its stopping
  rule, and the checks of the screening
- `intercept::Bool`: whether the path has intercepts
- `nobs::Int`, `task_nobs::Vector{Int}`, `task_weights::Vector`,
  `penalty_factors::Vector`, `row_norm::Symbol`, `alpha`: as in
  [`MultitaskLassoSolution`](@ref)

`coef` gives a vector with a coefficient matrix per level, with the intercepts
in the first row when the path has them (`coef(path, λ)` one matrix,
interpolated linearly in `λ`), `predict(path, Xs)` a vector with an `nₜ × L`
matrix of fitted values per task, a column per level, `nobs` the number of
observations, and `fit(MultitaskLassoPath, Xs, ys, ...)` is
`MultitaskLassoPath(Xs, ys, ...)`.
"""
struct MultitaskLassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  Bpath::Vector{SparseMatrixCSC{T, Int}}
  β0path::Vector{Vector{T}}
  activerows::Vector{Vector{Int}}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
  intercept::Bool
  nobs::Int
  task_nobs::Vector{Int}
  task_weights::Vector{T}
  penalty_factors::Vector{T}
  row_norm::Symbol
  alpha::T
end

MultitaskLassoPath{T}(λpath, Bpath, β0path, activerows, iterations, converged, kkt_checks, intercept, nobs,
                     task_nobs, task_weights, penalty_factors) where {T} =
  MultitaskLassoPath{T}(λpath, Bpath, β0path, activerows, iterations, converged, kkt_checks, intercept, nobs,
                       task_nobs, task_weights, penalty_factors, :l2, zero(T))
MultitaskLassoPath(λpath::Vector{T}, args::Vararg{Any, 11}) where {T<:AbstractFloat} =
  MultitaskLassoPath{T}(λpath, args...)

function MultitaskLassoPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                            λpath::AbstractVector{<:Real}, options::CDOptions = CDOptions(); intercept::Bool = true,
                            standardize_x::Bool = true, task_weights = :equal, penalty_factors = nothing,
                            rng::AbstractRNG = default_rng(), screening::Bool = true, max_rows = Inf,
                            row_norm::Symbol = :l2, alpha::Real = 0, kwargs...)
  _no_keywords(kwargs)
  _multitask_path(Xs, ys, λpath, options; intercept, standardize_x, task_weights, penalty_factors, rng, screening,
                  max_rows, row_norm, alpha)
end

function MultitaskLassoPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                            options::CDOptions = CDOptions(); nλ::Integer = 100,
                            λminratio::Union{Nothing, Real} = nothing, intercept::Bool = true,
                            standardize_x::Bool = true, task_weights = :equal, penalty_factors = nothing,
                            rng::AbstractRNG = default_rng(), screening::Bool = true, max_rows = Inf,
                            row_norm::Symbol = :l2, alpha::Real = 0, kwargs...)
  _no_keywords(kwargs)
  _multitask_path(Xs, ys, nothing, options; nλ, λminratio, intercept, standardize_x, task_weights, penalty_factors,
                  rng, screening, max_rows, row_norm, alpha)
end

# A path at λpath, or an automatic grid for nothing. With the default penalty
# and shared = false, shared designs are stacked too.
function _multitask_path(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                         λpath::Union{Nothing, AbstractVector{<:Real}}, options::CDOptions = CDOptions();
                         nλ::Integer = 100, λminratio::Union{Nothing, Real} = nothing, intercept::Bool = true,
                         standardize_x::Bool = true, task_weights = :equal, penalty_factors = nothing,
                         rng::AbstractRNG = default_rng(), screening::Bool = true, max_rows = Inf,
                         shared::Bool = true, row_norm::Symbol = :l2, alpha::Real = 0)
  _check_row_penalty(row_norm, alpha)
  if λpath === nothing
    nλ >= 1 || throw(ArgumentError("nλ must be positive"))
    λminratio === nothing || 0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  else
    all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  end
  d = _MultitaskData(Xs, ys; intercept, standardize_x, task_weights, shared,
                     stacked = row_norm === :l2 && iszero(alpha))
  T = eltype(d.S)
  p, q = size(d.S)
  ω = _penalty_factors(T, penalty_factors, p)
  row_norm === :l2 && iszero(alpha) ||
    return _row_path(d, ω, λpath, options; nλ, λminratio, row_norm, alpha, rng, screening, max_rows)
  m, x, groups, w = _multitask_problem(d, ω)
  g = GroupNormL2(one(T), groups, w)
  λs = if λpath === nothing
    all(>(0), ω) ||
      throw(ArgumentError("an automatic grid of penalty levels needs positive penalty factors; give λpath to leave predictors unpenalized."))
    ratio = T(something(λminratio, d.N < p * q ? 0.01 : 1e-4))
    λmax = lambda_max(m, g, x) * (1 + sqrt(eps(T)))
    nλ == 1 ? [λmax] : [λmax * ratio^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  else
    convert(Vector{T}, λpath)
  end
  path = solve_path(m, g, λs, _solver(options; rng); x, screening, maxnnz = max_rows, _gapkw(options)...)
  Bpath = [_task_coefficients(d, θ) for θ in path.solutions]
  MultitaskLassoPath{T}(path.λ, Bpath, [_task_intercepts(d, B) for B in Bpath], _activerows.(Bpath), path.passes,
                        path.converged, path.kkt_checks, intercept, d.N, d.nobs, d.c, ω)
end

"""
    CVMultitaskLassoPath

Result returned by [`CVMultitaskLassoPath`](@ref)`(Xs, ys)`.

# Fields
- `path::MultitaskLassoPath`: the path on all the data
- `λpath::Vector`: its levels, the grid of every fold
- `cvm::Vector`, `cvse::Vector`: the cross-validated prediction error at each
  level, and its standard error (`NaN` at the levels that no fold reaches, and
  `cvse` also where only one does)
- `cvm_task::Matrix`: the held-out mean squared error of each task (a row) at
  each level (a column), `NaN` where no fold reaches the level
- `nrows::Vector{Int}`: the number of nonzero rows of `path` at each level
- `λmin`, `λ1se`: among the levels reached and converged in every fold, the
  level with the smallest `cvm`, and the largest level whose `cvm` is within
  one standard error of it
- `folds::Vector{Vector{Int}}`: the fold of each observation, a vector per task
- `task_weights::Vector`: the task weights `cₜ` of the path on all the data
"""
struct CVMultitaskLassoPath{T<:AbstractFloat}
  path::MultitaskLassoPath{T}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  cvm_task::Matrix{T}
  nrows::Vector{Int}
  λmin::T
  λ1se::T
  folds::Vector{Vector{Int}}
  task_weights::Vector{T}
end

"""
    CVMultitaskLassoPath(Xs, ys, options = CDOptions(); nfolds = 10, folds = nothing,
                         rng = Random.default_rng(), threads = true, nλ = 100, λminratio, kwargs...)

K-fold cross-validation of the path of [`MultitaskLassoPath`](@ref) (with the
same options and keywords, such as `intercept`, `standardize_x`, `task_weights`,
`penalty_factors`, `row_norm`, `alpha` and `max_rows`), as [`CVLassoPath`](@ref)
does for one task:
the path on all the data gives the levels, each fold's path is computed on the
other folds at those levels, and the error on the held-out fold estimates the
prediction error at each level. The automatic grid needs finite λmax; sparse
group with `alpha > 0` permits zero factors.

The folds are drawn within each task: the observations of every task are
assigned to the `K = nfolds` folds in a balanced random way, drawn from `rng`
(`K` must not exceed the size of any task). `folds` gives them instead, a
vector of labels `1, …, K` per task, with every label used in every task. Each
fold's path refits the whole procedure on the observations of the other folds,
with the same `task_weights` argument: the centering, the scales, the weights of
`:observations` and the design caches are computed on them, so that nothing
leaks from the held-out fold. Every task must keep two training observations in
each fold with an intercept, and one without.

The error of fold `k` at a level is

    eₖ = ∑ₜ (cₜ / nₜ) ∑ᵢ (y_ti - ŷ_ti)² / wₖ,   wₖ = ∑ₜ cₜ mₜₖ / nₜ,

where the inner sum runs over the `mₜₖ` observations of task `t` in fold `k`, and
`cₜ` and `nₜ` are those of all the data. `cvm` is the average of the fold errors
weighted by the `wₖ`, and `cvse` its standard error,
`sqrt(∑ₖ wₖ (eₖ - cvm)² / ∑ₖ wₖ / (K - 1))`; with one task they are those of
[`CVLassoPath`](@ref). `cvm_task` holds the held-out mean squared errors `MSEₜ`
of the tasks, whose folds are weighted by the `mₜₖ`; at the levels that every
fold reaches, `cvm = ∑ₜ cₜ MSEₜ / ∑ₜ cₜ`, the held-out counterpart of the loss.
`λmin` minimizes `cvm`, and `λ1se` is the largest level whose `cvm` is within
`cvse` of the minimum (the one-standard-error rule). A fold path that stops
early (`max_rows`) leaves the remaining levels out of the averages;
nonconverged fold levels are treated the same way and are never selected. Levels
that no fold reaches have `NaN` in `cvm`, `cvse` and `cvm_task`. `λmin` and
`λ1se` are chosen among the levels that every fold reaches and converges at.

With `threads`, the folds run in parallel on the available threads; each fold
draws its random coordinate orders from a generator seeded from `rng`, so that
the result does not depend on the number of threads.

Returns a [`CVMultitaskLassoPath`](@ref); `coef(cv)` and `predict(cv, Xs)` use
the solution at `λ1se`, or at `λmin` with `λ = :λmin`, and
`fit(CVMultitaskLassoPath, Xs, ys, ...)` is `CVMultitaskLassoPath(Xs, ys, ...)`.
"""
function CVMultitaskLassoPath(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
                              options::CDOptions = CDOptions(); nfolds::Integer = 10,
                              folds::Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}} = nothing,
                              rng::AbstractRNG = default_rng(), threads::Bool = true, nλ::Integer = 100,
                              λminratio::Union{Nothing, Real} = nothing, intercept::Bool = true,
                              row_norm::Symbol = :l2, alpha::Real = 0, kwargs...)
  _check_row_penalty(row_norm, alpha)
  n, _ = _task_sizes(Xs, ys, intercept)
  q = length(n)
  fl = folds === nothing ? [_folds(rng, nt, nfolds) for nt in n] : _task_folds(folds, n)
  K = maximum(maximum, fl)
  mt = [count(==(k), fl[t]) for t in 1:q, k in 1:K]
  for t in 1:q, k in 1:K
    n[t] - mt[t, k] >= 1 + intercept ||
      throw(ArgumentError("fold $k leaves $(n[t] - mt[t, k]) training observations in task $t, but a fit$(intercept ? " with an intercept" : "") needs at least $(1 + intercept)."))
  end

  seeds = rand(rng, UInt64, K + 1)
  path = MultitaskLassoPath(Xs, ys, options; nλ, λminratio, intercept, row_norm, alpha,
                            rng = Xoshiro(seeds[end]), kwargs...)
  λs = path.λpath
  T, L = eltype(λs), length(λs)
  sse = fill(T(NaN), q, K, L)
  function fold!(k)
    train = [findall(!=(k), fl[t]) for t in 1:q]
    test = [findall(==(k), fl[t]) for t in 1:q]
    fp = MultitaskLassoPath([Xs[t][train[t], :] for t in 1:q], [ys[t][train[t]] for t in 1:q], λs, options;
                            intercept, row_norm, alpha, rng = Xoshiro(seeds[k]), kwargs...)
    Ŷ = predict(fp, [Xs[t][test[t], :] for t in 1:q])
    for t in 1:q, j in axes(Ŷ[t], 2)
      fp.converged[j] || continue
      sse[t, k, j] = sum(i -> abs2(ys[t][test[t][i]] - Ŷ[t][i, j]), eachindex(test[t]))
    end
  end
  _runfolds(fold!, K, threads)

  c = path.task_weights
  wk = T[sum(c[t] * mt[t, k] / n[t] for t in 1:q) for k in 1:K]
  errs = T[isnan(sse[1, k, j]) ? T(NaN) : sum(c[t] / n[t] * sse[t, k, j] for t in 1:q) / wk[k] for k in 1:K, j in 1:L]
  cvm, cvse, jmin, j1se = _cvsummary(errs, wk)
  cvm_task = fill(T(NaN), q, L)
  for t in 1:q, j in 1:L
    ok = [k for k in 1:K if !isnan(sse[t, k, j])]
    isempty(ok) || (cvm_task[t, j] = sum(sse[t, ok, j]) / sum(mt[t, ok]))
  end
  CVMultitaskLassoPath{T}(path, λs, cvm, cvse, cvm_task, length.(path.activerows), λs[jmin], λs[j1se], fl, c)
end

# the folds given, a vector of labels 1, …, K per task, after checks
function _task_folds(folds::AbstractVector, n::Vector{Int})
  q = length(n)
  length(folds) == q ||
    throw(DimensionMismatch("folds must have a vector of labels per task, but it has $(length(folds)) for $q tasks"))
  fl = [collect(Int, f) for f in folds]
  for t in 1:q
    length(fl[t]) == n[t] ||
      throw(DimensionMismatch("folds[$t] has $(length(fl[t])) labels, but task $t has $(n[t]) observations"))
  end
  K = maximum(maximum, fl)
  K >= 2 && all(f -> all(l -> 1 <= l <= K, f) && all(k -> any(==(k), f), 1:K), fl) ||
    throw(ArgumentError("the fold labels must be 1, …, K for some K ≥ 2, each used at least once in every task"))
  fl
end
