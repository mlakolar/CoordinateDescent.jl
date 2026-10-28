######################################################################
#
#   The multi-response lasso
#
######################################################################

# the multi-response least-squares model of B, the rows of B as entries of
# vec(B), a zero iterate over vec(B) with those groups, and the loadings
function _multi_problem(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, intercept::Bool, standardize_x::Bool) where {T}
  n, p = size(X)
  q = size(Y, 2)
  size(Y, 1) == n || throw(DimensionMismatch("X has $n rows and Y has $(size(Y, 1))"))
  _check_finite(X, "X"); _check_finite(Y, "Y")
  groups = [collect(k:p:p*q) for k in 1:p]
  x = AtomIterate(T, p * q, groups)
  m = ProximalSolvers.cdmodel((1 / T(n)) * LeastSquaresLoss(Y, X), x; intercept)
  m, groups, x, (standardize_x ? _loadings(X, intercept) : ones(T, p))
end

_activerows(B::AbstractMatrix) = [k for k in axes(B, 1) if any(!iszero, view(B, k, :))]

"""
    MultiLassoSolution

Result returned by [`multi_lasso`](@ref).

# Fields
- `B::SparseMatrixCSC`: the `p × q` matrix of coefficients, a column per response
- `intercept::Vector`: the unpenalized intercepts of the responses, zero unless
  `intercept = true`
- `λ`: the penalty level
- `activerows::Vector{Int}`: the rows with a nonzero coefficient; entries of
  a selected row can still be zero
- `iterations::Int`, `converged::Bool`: the passes of coordinate descent over
  the rows and whether it met its stopping rule
- `hasintercept::Bool`, `nobs::Int`: whether the solver fitted intercepts, and
  the number of observations
- `row_norm::Symbol`, `alpha`: the row norm and share of the entrywise penalty

[`coef`](@ref) (a `(p + 1) × q` matrix, the intercepts in the first row),
[`predict`](@ref) (an `n × q` matrix), [`nobs`](@ref) and [`dof`](@ref) apply
to it.
"""
struct MultiLassoSolution{T}
  B::SparseMatrixCSC{T, Int}
  intercept::Vector{T}
  λ::T
  activerows::Vector{Int}
  iterations::Int
  converged::Bool
  hasintercept::Bool
  nobs::Int
  row_norm::Symbol
  alpha::T
end

MultiLassoSolution{T}(B, intercept, λ, activerows, iterations, converged, hasintercept, nobs) where {T} =
  MultiLassoSolution{T}(B, intercept, λ, activerows, iterations, converged, hasintercept, nobs, :l2, zero(T))
MultiLassoSolution(B::SparseMatrixCSC{T, Int}, args::Vararg{Any, 7}) where {T} = MultiLassoSolution{T}(B, args...)

"""
    multi_lasso(X, Y, λ, options = CDOptions(); intercept = true, standardize_x = true,
                row_norm = :l2, alpha = 0, rng = Random.default_rng())

The multi-response lasso of the `n × q` responses `Y` on `X`,

    minimize  ‖Y - 1β₀' - XB‖²_F / (2n) + λ ∑ⱼ h((sⱼ B[j, t])ₜ)

where `h(u) = alpha * ‖u‖₁ + (1 - alpha) * ‖u‖₂` with `row_norm = :l2`, or
`h(u) = ‖u‖∞` with `row_norm = :linf` (which requires `alpha = 0`). `alpha`
must lie in `[0, 1]`. The default row ℓ1/ℓ2 penalty is glmnet's `mgaussian`
family. Row selection allows zero entries within a selected row.
The intercepts `β₀` are not penalized, and they are zero unless `intercept` is
true. The scales `sⱼ` are the population standard deviations of the columns,
centered with an intercept, with `standardize_x`, and one otherwise.
Coefficients are returned on the original scale. With one response all three
penalties reduce to [`lasso`](@ref) with matching standardization.

It is solved by block coordinate descent on the rows of `B`, with `options`.
The default uses ProximalSolvers' multi-response least-squares model; the new
penalties use row coordinate descent on the shared design, without stacking.
Its steps are exact, and orders are drawn from `rng`;
with `options.warm_start` false the solution is computed along
`options.num_steps` levels from λmax. Returns a [`MultiLassoSolution`](@ref).
See also [`MultiLassoPath`](@ref) and [`CVMultiLassoPath`](@ref).
"""
function multi_lasso(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, λ::T, options::CDOptions = CDOptions();
                    intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                    row_norm::Symbol = :l2, alpha::Real = 0, rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  _check_keywords(CDOptions, (), kwargs)
  _check_row_penalty(row_norm, alpha)
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  if row_norm !== :l2 || !iszero(alpha)
    sol = multitask_lasso(fill(X, size(Y, 2)), [Y[:, t] for t in axes(Y, 2)], λ, options;
                          intercept, standardize_x, row_norm, alpha, rng)
    return MultiLassoSolution{T}(sol.B, sol.intercept, λ, sol.activerows, sol.iterations, sol.converged,
                                 intercept, size(X, 1), row_norm, T(alpha))
  end
  p, q = size(X, 2), size(Y, 2)
  m, groups, x, ω = _multi_problem(X, Y, intercept, standardize_x)
  r = _solve!(x, m, GroupNormL2(λ, groups, ω), options; rng)
  B = sparse(reshape(Vector(x), p, q))
  MultiLassoSolution{T}(B, ProximalSolvers.intercept(m, x), λ, _activerows(B), r.passes, r.converged, intercept,
                        size(X, 1))
end

function multi_lasso(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, λ::Real, options::CDOptions = CDOptions();
                    kwargs...)
  T = float(promote_type(eltype(X), eltype(Y)))
  multi_lasso(convert(AbstractMatrix{T}, X), convert(AbstractMatrix{T}, Y), T(λ), options; kwargs...)
end

"""
    MultiLassoPath(X, Y, options = CDOptions(); nλ = 100, λminratio, kwargs...)
    MultiLassoPath(X, Y, λpath, options = CDOptions(); intercept = true, standardize_x = true,
                   screening = true, max_rows = Inf, row_norm = :l2, alpha = 0,
                   rng = Random.default_rng())

Solutions `β0path[k]` and `Bpath[k]` of the multi-response lasso of
[`multi_lasso`](@ref) along a decreasing sequence `λpath`, each one warm started
from the previous solution, by ProximalSolvers' `solve_path` for the default
penalty and by row coordinate descent for the new penalties. `row_norm` and
`alpha` record the penalty on the path; the old positional constructor fills
`:l2` and zero.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`.

With `screening`, each λ is solved first over the rows kept by the sequential
strong rule, and the optimality conditions of the others are checked
afterwards (`kkt_checks[k]` times at `λpath[k]`). The path stops early after the
first solution with more than `max_rows` nonzero rows; the returned `λpath`
then contains only the values that were solved for. `activerows[k]` lists the
selected predictors at `λpath[k]`. `coef` and `predict` give a vector with a
matrix per level (`coef(path, λ)` one matrix, interpolated in `λ`), `nobs` the
number of observations, and `fit(MultiLassoPath, X, Y, ...)` is
`MultiLassoPath(X, Y, ...)`.
"""
struct MultiLassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  Bpath::Vector{SparseMatrixCSC{T, Int}}
  β0path::Vector{Vector{T}}
  activerows::Vector{Vector{Int}}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
  intercept::Bool
  nobs::Int
  row_norm::Symbol
  alpha::T
end

MultiLassoPath{T}(λpath, Bpath, β0path, activerows, iterations, converged, kkt_checks, intercept, nobs) where {T} =
  MultiLassoPath{T}(λpath, Bpath, β0path, activerows, iterations, converged, kkt_checks, intercept, nobs, :l2, zero(T))
MultiLassoPath(λpath::Vector{T}, args::Vararg{Any, 8}) where {T<:AbstractFloat} = MultiLassoPath{T}(λpath, args...)

function MultiLassoPath(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, λpath::Vector{T}, options::CDOptions = CDOptions();
                        intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing, screening::Bool = true,
                        max_rows = Inf, row_norm::Symbol = :l2, alpha::Real = 0,
                        rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  _check_keywords(CDOptions, (), kwargs)
  _check_row_penalty(row_norm, alpha)
  if row_norm !== :l2 || !iszero(alpha)
    path = MultitaskLassoPath(fill(X, size(Y, 2)), [Y[:, t] for t in axes(Y, 2)], λpath, options;
                              intercept, standardize_x, row_norm, alpha, rng, screening, max_rows)
    return _multi_row_path(path, size(X, 1))
  end
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  p, q = size(X, 2), size(Y, 2)
  m, groups, x, ω = _multi_problem(X, Y, intercept, standardize_x)
  β0path = Vector{T}[]
  path = solve_path(m, GroupNormL2(one(T), groups, ω), λpath, _solver(options; rng); x, screening, maxnnz = max_rows,
                    _gapkw(options)...,
                    callback = (k, λ, x, m) -> (push!(β0path, ProximalSolvers.intercept(m, x)); false))
  Bpath = [sparse(reshape(Vector(θ), p, q)) for θ in path.solutions]
  MultiLassoPath{T}(path.λ, Bpath, β0path, _activerows.(Bpath), path.passes, path.converged, path.kkt_checks, intercept,
                    size(X, 1))
end

function MultiLassoPath(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, options::CDOptions = CDOptions();
                        nλ::Integer = 100, λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                        intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                        row_norm::Symbol = :l2, alpha::Real = 0, rng::AbstractRNG = default_rng(),
                        screening::Bool = true, max_rows = Inf, kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  _check_keywords(CDOptions, (), kwargs)
  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  _check_row_penalty(row_norm, alpha)
  if row_norm !== :l2 || !iszero(alpha)
    path = MultitaskLassoPath(fill(X, size(Y, 2)), [Y[:, t] for t in axes(Y, 2)], options;
                              nλ, λminratio, intercept, standardize_x, row_norm, alpha, rng, screening, max_rows)
    return _multi_row_path(path, size(X, 1))
  end
  m, groups, x, ω = _multi_problem(X, Y, intercept, standardize_x)
  λmax = lambda_max(m, GroupNormL2(one(T), groups, ω), x) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  MultiLassoPath(X, Y, λpath, options; intercept, standardize_x, rng, screening, max_rows)
end

function MultiLassoPath(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, λpath::AbstractVector{<:Real},
                        options::CDOptions = CDOptions(); kwargs...)
  T = float(promote_type(eltype(X), eltype(Y)))
  MultiLassoPath(convert(AbstractMatrix{T}, X), convert(AbstractMatrix{T}, Y), convert(Vector{T}, λpath), options; kwargs...)
end

function MultiLassoPath(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, options::CDOptions = CDOptions(); kwargs...)
  T = float(promote_type(eltype(X), eltype(Y)))
  MultiLassoPath(convert(AbstractMatrix{T}, X), convert(AbstractMatrix{T}, Y), options; kwargs...)
end

# A path from the row solver, with observations shared by all responses.
_multi_row_path(path, n) = MultiLassoPath(path.λpath, path.Bpath, path.β0path, path.activerows, path.iterations,
                                        path.converged, path.kkt_checks, path.intercept, n, path.row_norm, path.alpha)

"""
    CVMultiLassoPath(X, Y, options = CDOptions(); nfolds = 10, folds = nothing,
                     rng = Random.default_rng(), threads = true, nλ = 100, λminratio, kwargs...)

Cross-validation of [`MultiLassoPath`](@ref), which minimizes

    ‖Y - 1β₀' - XB‖²_F / (2n) + λ ∑ⱼ hⱼ((sⱼ B[j, t])ₜ).

The row penalty is selected by `row_norm` and `alpha` as in [`multi_lasso`](@ref).
Observations are assigned to `nfolds` balanced folds, shared by all responses,
from `rng`, or to the labels in `folds`. Each training fold refits the centering,
scales and coefficients at the full-data path's levels. The held-out error of
fold k is the mean squared error over its observations and responses.

# Fields
- `path::MultiLassoPath`, `λpath::Vector`: the full-data path and its levels
- `cvm::Vector`, `cvse::Vector`: the fold-size-weighted error and its standard
  error, as in [`CVLassoPath`](@ref)
- `cvm_response::Matrix`: the held-out mean squared error of each response (a
  row) at each level (a column)
- `nrows::Vector{Int}`: the number of selected rows of the full-data path
- `λmin`, `λ1se`: the minimum-error and one-standard-error levels, among those
  reached and converged in every fold
- `folds::Vector{Int}`: the fold labels of the observations

Unreached or nonconverged fold levels have no error (`NaN`); they are never
selected. Each fold's generator is seeded from `rng`, independently of threads.
`coef(cv)` and `predict(cv, X)` use `λ1se`, or `λmin` with `λ = :λmin`.
"""
struct CVMultiLassoPath{T<:AbstractFloat}
  path::MultiLassoPath{T}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  cvm_response::Matrix{T}
  nrows::Vector{Int}
  λmin::T
  λ1se::T
  folds::Vector{Int}
end

function CVMultiLassoPath(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real},
                          options::CDOptions = CDOptions(); nfolds::Integer = 10,
                          folds::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                          rng::AbstractRNG = default_rng(), threads::Bool = true, nλ::Integer = 100,
                          λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                          intercept::Bool = true, kwargs...)
  n, q = size(Y)
  size(X, 1) == n || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $n"))
  n > 0 && q > 0 || throw(ArgumentError("Y must have at least one observation and response."))
  fl, K = _foldlabels(rng, n, nfolds, folds)
  all(k -> 1 <= k <= K, fl) || throw(ArgumentError("the fold labels must be 1, …, K."))
  mt = [count(==(k), fl) for k in 1:K]
  all(m -> n - m >= 1 + intercept, mt) ||
    throw(ArgumentError("each fold must leave at least $(1 + intercept) training observations."))
  seeds = rand(rng, UInt64, K + 1)
  path = MultiLassoPath(X, Y, options; nλ, λminratio, intercept, rng = Xoshiro(seeds[end]), kwargs...)
  λs = path.λpath
  T, L = eltype(λs), length(λs)
  sse = fill(T(NaN), q, K, L)
  function fold!(k)
    train, test = findall(!=(k), fl), findall(==(k), fl)
    fp = MultiLassoPath(X[train, :], Y[train, :], λs, options;
                        intercept, rng = Xoshiro(seeds[k]), kwargs...)
    Ŷ = predict(fp, X[test, :])
    for j in eachindex(Ŷ)
      fp.converged[j] || continue
      for t in 1:q
        sse[t, k, j] = sum(i -> abs2(Y[test[i], t] - Ŷ[j][i, t]), eachindex(test))
      end
    end
  end
  _runfolds(fold!, K, threads)
  errs = T[sum(view(sse, :, k, j)) / (q * mt[k]) for k in 1:K, j in 1:L]
  cvm, cvse, jmin, j1se = _cvsummary(errs, T.(mt))
  cvm_response = fill(T(NaN), q, L)
  for t in 1:q, j in 1:L
    ok = [k for k in 1:K if !isnan(sse[t, k, j])]
    isempty(ok) || (cvm_response[t, j] = sum(sse[t, ok, j]) / sum(mt[ok]))
  end
  CVMultiLassoPath{T}(path, λs, cvm, cvse, cvm_response, length.(path.activerows), λs[jmin], λs[j1se], fl)
end
