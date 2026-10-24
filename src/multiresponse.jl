######################################################################
#
#   The multi-response lasso
#
######################################################################

# the multi-response least-squares model of B, the rows of B as entries of
# vec(B), a zero iterate over vec(B) with those groups, and the loadings
function _multiProblem(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, intercept::Bool, standardizeX::Bool) where {T}
  n, p = size(X)
  q = size(Y, 2)
  size(Y, 1) == n || throw(DimensionMismatch("X has $n rows and Y has $(size(Y, 1))"))
  _check_finite(X, "X"); _check_finite(Y, "Y")
  groups = [collect(k:p:p*q) for k in 1:p]
  x = AtomIterate(T, p * q, groups)
  m = ProximalSolvers.cdmodel((1 / T(n)) * LeastSquaresLoss(Y, X), x; intercept)
  m, groups, x, (standardizeX ? _loadings(X, intercept) : ones(T, p))
end

_activerows(B::AbstractMatrix) = [k for k in axes(B, 1) if any(!iszero, view(B, k, :))]

"""
    MultiLassoSolution

Result returned by [`multiLasso`](@ref).

# Fields
- `B::SparseMatrixCSC`: the `p × q` matrix of coefficients, a column per response
- `intercept::Vector`: the unpenalized intercepts of the responses, zero unless
  `intercept = true`
- `λ`: the penalty level
- `activerows::Vector{Int}`: the predictors with nonzero coefficients, the same
  for all the responses
- `iterations::Int`, `converged::Bool`: the passes of coordinate descent over
  the rows and whether it met its stopping rule
- `hasintercept::Bool`, `nobs::Int`: whether the solver fitted intercepts, and
  the number of observations

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
end

"""
    multiLasso(X, Y, λ, options = CDOptions(); intercept = true, standardizeX = true)

The multi-response lasso of the `n × q` responses `Y` on `X`,

    minimize  ‖Y - 1β₀' - XB‖²_F / (2n) + λ ∑ₖ ωₖ ‖B[k, :]‖₂

which selects the same predictors for all the responses (glmnet's
`mgaussian` family). The intercepts `β₀` are not penalized, and they are zero
unless `intercept` is true. The loadings `ωₖ` are the standard deviations of
the columns, centered with an intercept, when `standardizeX` is true, and one
otherwise. With one response it is [`lasso`](@ref).

It is solved by block coordinate descent on the rows of `B` (ProximalSolvers'
multi-response least-squares model), whose steps are exact, with `options`;
with `options.warmStart` false the solution is computed along
`options.numSteps` levels from λmax. Returns a [`MultiLassoSolution`](@ref).
See also [`MultiLassoPath`](@ref).
"""
function multiLasso(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, λ::T, options::CDOptions = CDOptions();
                    intercept::Bool = true, standardizeX::Bool = true) where {T<:AbstractFloat}
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  p, q = size(X, 2), size(Y, 2)
  m, groups, x, ω = _multiProblem(X, Y, intercept, standardizeX)
  r = _solve!(x, m, GroupNormL2(λ, groups, ω), options)
  B = sparse(reshape(Vector(x), p, q))
  MultiLassoSolution{T}(B, ProximalSolvers.intercept(m, x), λ, _activerows(B), r.passes, r.converged, intercept,
                        size(X, 1))
end

function multiLasso(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, λ::Real, options::CDOptions = CDOptions();
                    kwargs...)
  T = float(promote_type(eltype(X), eltype(Y)))
  multiLasso(convert(AbstractMatrix{T}, X), convert(AbstractMatrix{T}, Y), T(λ), options; kwargs...)
end

"""
    MultiLassoPath(X, Y, options = CDOptions(); nλ = 100, λminratio, kwargs...)
    MultiLassoPath(X, Y, λpath, options = CDOptions(); intercept = true, standardizeX = true,
                   screening = true, max_rows = Inf)

Solutions `β0path[k]` and `Bpath[k]` of the multi-response lasso of
[`multiLasso`](@ref) along a decreasing sequence `λpath`, each one warm started
from the previous solution, by ProximalSolvers' `solve_path`.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`.

With `screening`, each λ is solved first over the rows kept by the sequential
strong rule, and the optimality conditions of the others are checked
afterwards (`kktChecks[k]` times at `λpath[k]`). The path stops early after the
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
  kktChecks::Vector{Int}
  intercept::Bool
  nobs::Int
end

function MultiLassoPath(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, λpath::Vector{T}, options::CDOptions = CDOptions();
                        intercept::Bool = true, standardizeX::Bool = true, screening::Bool = true,
                        max_rows = Inf) where {T<:AbstractFloat}
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  p, q = size(X, 2), size(Y, 2)
  m, groups, x, ω = _multiProblem(X, Y, intercept, standardizeX)
  β0path = Vector{T}[]
  path = solve_path(m, GroupNormL2(one(T), groups, ω), λpath, _solver(options); x, screening, maxnnz = max_rows,
                    callback = (k, λ, x, m) -> (push!(β0path, ProximalSolvers.intercept(m, x)); false))
  Bpath = [sparse(reshape(Vector(θ), p, q)) for θ in path.solutions]
  MultiLassoPath{T}(path.λ, Bpath, β0path, _activerows.(Bpath), path.passes, path.converged, path.kkt_checks, intercept,
                    size(X, 1))
end

function MultiLassoPath(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, options::CDOptions = CDOptions();
                        nλ::Integer = 100, λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                        intercept::Bool = true, standardizeX::Bool = true, kwargs...) where {T<:AbstractFloat}
  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  m, groups, x, ω = _multiProblem(X, Y, intercept, standardizeX)
  λmax = lambda_max(m, GroupNormL2(one(T), groups, ω), x) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  MultiLassoPath(X, Y, λpath, options; intercept, standardizeX, kwargs...)
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
