######################################################################
#
#   The sparse group lasso
#
######################################################################

"""
    SparseGroupLassoSolution

Result returned by [`sparse_group_lasso`](@ref).

# Fields
- `x::SparseVector`: the coefficients `β`, on the scale of the data
- `intercept`: the unpenalized intercept `β₀`, zero unless `intercept = true`
- `λ`, `alpha`: the penalty level and the mix of the ℓ1 and group terms
- `groups::Vector{Vector{Int}}`, `groupweights::Vector`: the groups and their
  weights `wₖ`
- `activegroups::Vector{Int}`: the groups with nonzero coefficients
- `scales`: `nothing`, or the scales `s` of the columns when the penalty
  applies to `s .* x` (`standardize_x`)
- `iterations::Int`, `converged::Bool`: the passes of coordinate descent and
  whether it met its stopping rule
- `hasintercept::Bool`, `nobs::Int`: whether the solver fitted an intercept, and
  the number of observations

[`coef`](@ref), [`predict`](@ref), [`nobs`](@ref) and [`dof`](@ref) apply to it.
"""
struct SparseGroupLassoSolution{T}
  x::SparseVector{T, Int}
  intercept::T
  λ::T
  alpha::T
  groups::Vector{Vector{Int}}
  groupweights::Vector{T}
  activegroups::Vector{Int}
  scales::Union{Nothing, Vector{T}}
  iterations::Int
  converged::Bool
  hasintercept::Bool
  nobs::Int
end

"""
    sparse_group_lasso(X, y, λ, groups, options = CDOptions(); alpha = 0.95, intercept = true, standardize_x = true,
                     groupweights = sqrt.(length.(groups)), rng = Random.default_rng())

The sparse group lasso of Simon, Friedman, Hastie and Tibshirani (2013),

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ (α ∑ⱼ |β̃ⱼ| + (1 - α) ∑ₖ wₖ ‖β̃_Gₖ‖₂),

which selects groups, as the group lasso does, and also variables within the
selected groups, for disjoint groups `Gₖ` that cover the columns of `X`. With
`standardize_x`, `β̃ = s .* β` are the coefficients of the columns divided by their
standard deviations `s` (centered with an intercept); otherwise `β̃ = β`. R's
sparsegl and SGL scale the columns to unit Euclidean norm instead, `√n` times
the standard deviation, so their level `λ` is `√n λ` here. The coefficients returned are those of the columns of `X`. `alpha`
mixes the two terms: `alpha = 1` is the lasso and `alpha = 0` the group lasso on
the columns as given (`group_lasso(...; standardize = :none)`); the default 0.95
is SGL's. The weights default to `wₖ = √|Gₖ|`, and a zero weight with
`alpha = 0` leaves a group unpenalized. The intercept `β₀` is not penalized, and
it is zero unless `intercept` is true.

It is solved by block coordinate descent on the groups (ProximalSolvers' CD on
an `AtomIterate`, whose block steps are the exact proximal operator of the
penalty plus a quadratic bound), with `options`; with `options.warm_start`
false the solution is computed along `options.num_steps` levels from λmax. The
random order is drawn from `rng`. Unlike [`group_lasso`](@ref), the groups are
not orthonormalized: the ℓ1 term is not invariant to a change of basis within
a group. Returns a [`SparseGroupLassoSolution`](@ref). See also
[`SparseGroupLassoPath`](@ref).

Reference: N. Simon, J. Friedman, T. Hastie and R. Tibshirani (2013). A
sparse-group lasso. *Journal of Computational and Graphical Statistics* 22(2),
231–245.
"""
function sparse_group_lasso(X::AbstractMatrix{T}, y::AbstractVector{T}, λ::T, groups::AbstractVector,
                          options::CDOptions = CDOptions(); alpha::Real = 0.95, intercept::Bool = true,
                          standardize_x::Union{Nothing, Bool}=nothing, groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                          rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  _check_keywords(CDOptions, (), kwargs)
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  m, g1, x, gs, w, s = _sparse_group_problem(X, y, groups, T(alpha), intercept, standardize_x, groupweights)
  r = _solve_levels!(x, m, _atlevel(g1, λ), options; rng)
  β0, β = ProximalSolvers.intercept(m, x), _unscaledcoef(Vector(x), s)
  SparseGroupLassoSolution{T}(β, β0, λ, T(alpha), gs, w, _activegroups(gs, β), s, r.passes, r.converged, intercept,
                              size(X, 1))
end

function sparse_group_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real, groups::AbstractVector,
                          options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  sparse_group_lasso(Xf, yf, T(λ), groups, options; kwargs...)
end

# The least-squares model on the columns divided by their scales s (nothing
# without standardize_x), the penalty at level one,
# Scaled(SparseGroupNormL2(α, 1 - α), 1), whose level is the scale in
# ProximalSolvers' paths, a zero iterate with the groups, the groups and their
# weights, and s
function _sparse_group_problem(X::AbstractMatrix{T}, y::AbstractVector{T}, groups, α::T, intercept::Bool,
                             standardize_x::Bool, groupweights) where {T}
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  0 <= α <= 1 || throw(ArgumentError("alpha must be in [0, 1], but it is $α."))
  _check_finite(X, "X"); _check_finite(y, "y")
  gs = _check_groups(groups, p)
  w = groupweights === nothing ? T.(sqrt.(length.(gs))) : convert(Vector{T}, groupweights)
  length(w) == length(gs) || throw(DimensionMismatch("there are $(length(gs)) groups and $(length(w)) group weights"))
  all(>=(0), w) || throw(ArgumentError("the group weights must be nonnegative"))
  s = standardize_x ? _scales(X, intercept, nothing) : nothing
  m = CDLeastSquaresLoss(y, _scale_columns(X, s); intercept)
  m, Scaled(SparseGroupNormL2(α, 1 - α, gs, w), one(T)), AtomIterate(T, p, gs), gs, w, s
end

_atlevel(g::Scaled{<:SparseGroupNormL2}, λ) = Scaled(g.f, λ)

_unscaledcoef(b::Vector, ::Nothing) = sparse(b)
_unscaledcoef(b::Vector, s::Vector) = sparse(b ./ s)

"""
    SparseGroupLassoPath(X, y, groups, options = CDOptions(); nλ = 100, λminratio, kwargs...)
    SparseGroupLassoPath(X, y, λpath, groups, options = CDOptions(); alpha = 0.95, intercept = true,
                         standardize_x = true, groupweights = sqrt.(length.(groups)), screening = true,
                         max_groups = Inf, rng = Random.default_rng())

Solutions `β0path[k]` and `βpath[k]` of the sparse group lasso of
[`sparse_group_lasso`](@ref) along a decreasing sequence `λpath`, each one warm
started from the previous solution, by ProximalSolvers' `solve_path`, at the
fixed mix `alpha`.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`. A group at zero is
optimal while `‖soft(∇_G, λα)‖₂ ≤ λ(1 - α) w_G` (`soft` the soft-thresholding of
its gradient); `λmax` is the largest root of these conditions at zero, and the
sequential strong rule (with `screening`) and its checks compare each group's
root with `λ`.

The path stops early after the first solution with more than `max_groups`
nonzero groups; the returned `λpath` then contains only the values that were
solved for. `activegroups[k]` lists the nonzero groups at `λpath[k]`.
`coef`, `predict` and `nobs` apply to the path, as for [`LassoPath`](@ref), and
`fit(SparseGroupLassoPath, X, y, ...)` is `SparseGroupLassoPath(X, y, ...)`.
"""
struct SparseGroupLassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseVector{T, Int}}
  β0path::Vector{T}
  activegroups::Vector{Vector{Int}}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
  intercept::Bool
  alpha::T
  groups::Vector{Vector{Int}}
  groupweights::Vector{T}
  scales::Union{Nothing, Vector{T}}
  nobs::Int
end

function SparseGroupLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, λpath::Vector{T}, groups::AbstractVector,
                              options::CDOptions = CDOptions(); alpha::Real = 0.95, intercept::Bool = true,
                              standardize_x::Union{Nothing, Bool}=nothing, groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                              screening::Bool = true, max_groups = Inf,
                              rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  _check_keywords(CDOptions, (), kwargs)
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  m, g1, x, gs, w, s = _sparse_group_problem(X, y, groups, T(alpha), intercept, standardize_x, groupweights)
  β0path = T[]
  path = solve_path(m, g1, λpath, _solver(options; rng); x, screening, maxnnz = max_groups, _gapkw(options)...,
                    callback = (k, λ, x, m) -> (push!(β0path, ProximalSolvers.intercept(m, x)); false))
  βpath = [_unscaledcoef(Vector(θ), s) for θ in path.solutions]
  SparseGroupLassoPath{T}(path.λ, βpath, β0path, [_activegroups(gs, β) for β in βpath], path.passes, path.converged,
                          path.kkt_checks, intercept, T(alpha), gs, w, s, size(X, 1))
end

function SparseGroupLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, groups::AbstractVector,
                              options::CDOptions = CDOptions(); nλ::Integer = 100,
                              λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4), alpha::Real = 0.95,
                              intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                              groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                              kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  m, g1, x, _, _, _ = _sparse_group_problem(X, y, groups, T(alpha), intercept, standardize_x, groupweights)
  λmax = lambda_max(m, g1, x) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  SparseGroupLassoPath(X, y, λpath, groups, options; alpha, intercept, standardize_x, groupweights, kwargs...)
end

function SparseGroupLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λpath::AbstractVector{<:Real},
                              groups::AbstractVector, options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  SparseGroupLassoPath(Xf, yf, convert(Vector{T}, λpath), groups, options; kwargs...)
end

function SparseGroupLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, groups::AbstractVector,
                              options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  SparseGroupLassoPath(Xf, yf, groups, options; kwargs...)
end
