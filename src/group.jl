######################################################################
#
#   The group lasso
#
######################################################################

# The groups as the estimators take them: disjoint and covering 1:p
function _check_groups(groups, p::Integer)
  seen = falses(p)
  for G in groups
    for j in G
      1 <= j <= p || throw(ArgumentError("group index $j is not a column of X, which has $p columns"))
      seen[j] && throw(ArgumentError("column $j is in two groups; the groups must be disjoint"))
      seen[j] = true
    end
  end
  all(seen) || throw(ArgumentError("columns $(findall(!, seen)) are in no group; give each its own group (a zero group weight leaves it unpenalized)"))
  [collect(Int, G) for G in groups]
end

# The design on which a group lasso is fitted, and the map from its
# coefficients θ back to the coefficients β of the columns of X. With
# orthonormal groups, each group of centered columns X̃_G (centered only with an
# intercept) is factored by a pivoted QR as X̃_G P = Q R; its first r columns
# span it, r its numerical rank, and Z_G = √n Q[:, 1:r] has Z_G'Z_G / n = I, so
# that X̃_G[:, P[1:r]] = Z_G R̃ with R̃ = R[1:r, 1:r] / √n and β_G = R̃⁻¹ θ_G on
# those columns, zero on the others. A group of rank zero (zero or, with an
# intercept, constant columns) is left out of the fit. Without
# orthonormalization, Z is X and the groups are the given ones.
struct _GroupDesign{T, MZ<:AbstractMatrix{T}}
  Z::MZ
  zgroups::Vector{Vector{Int}}             # the groups of the fit, as columns of Z
  fitted::Vector{Int}                      # the group of each group of the fit
  cols::Vector{Vector{Int}}                # the columns of X that each group of the fit spans
  R::Vector{UpperTriangular{T, Matrix{T}}} # R̃, empty without orthonormalization
end

function _group_design(X::AbstractMatrix{T}, groups::Vector{Vector{Int}}, intercept::Bool,
                       orthonormal::Bool) where {T}
  orthonormal || return _GroupDesign{T, typeof(X)}(X, groups, collect(1:length(groups)), groups,
                                                   UpperTriangular{T, Matrix{T}}[])
  n = size(X, 1)
  xbar = intercept ? _column_means(X) : zeros(T, size(X, 2))
  blocks, zgroups, fitted, cols, R = Matrix{T}[], Vector{Int}[], Int[], Vector{Int}[], UpperTriangular{T, Matrix{T}}[]
  off = 0
  for (k, G) in enumerate(groups)
    F = qr(X[:, G] .- transpose(xbar[G]), ColumnNorm())
    d = abs.(diag(F.R))
    r = isempty(d) ? 0 : count(>(max(n, length(G)) * eps(T) * d[1]), d)
    r == 0 && continue
    push!(blocks, sqrt(T(n)) .* Matrix(F.Q)[:, 1:r])
    push!(R, UpperTriangular(F.R[1:r, 1:r] ./ sqrt(T(n))))
    push!(cols, G[F.p[1:r]])
    push!(zgroups, collect(off+1:off+r))
    push!(fitted, k)
    off += r
  end
  Z = isempty(blocks) ? zeros(T, n, 0) : reduce(hcat, blocks)
  _GroupDesign{T, Matrix{T}}(Z, zgroups, fitted, cols, R)
end

# β, as a SparseVector of length p, from the coefficients θ of the fit
function _coefficients(d::_GroupDesign{T}, θ::AbstractVector{T}, p::Integer) where {T}
  ind, val = Int[], T[]
  for (k, zg) in enumerate(d.zgroups)
    θG = θ[zg]
    all(iszero, θG) && continue
    append!(ind, d.cols[k])
    append!(val, isempty(d.R) ? θG : d.R[k] \ θG)
  end
  sparsevec(ind, val, p)
end

# the groups whose coefficients are not all zero
_activegroups(groups, β::AbstractVector) = [k for (k, G) in enumerate(groups) if any(j -> !iszero(β[j]), G)]

# the least-squares model, the penalty with level one and a zero iterate of the
# fit, and the design
function _groupProblem(X::AbstractMatrix{T}, y::AbstractVector{T}, groups, intercept::Bool, standardize::Symbol,
                       groupweights) where {T}
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  standardize in (:orthonormal, :none) ||
    throw(ArgumentError("standardize must be :orthonormal or :none, but it is :$standardize."))
  gs = _check_groups(groups, p)
  w = groupweights === nothing ? T.(sqrt.(length.(gs))) : convert(Vector{T}, groupweights)
  length(w) == length(gs) || throw(DimensionMismatch("there are $(length(gs)) groups and $(length(w)) group weights"))
  all(>=(0), w) || throw(ArgumentError("the group weights must be nonnegative"))
  d = _group_design(X, gs, intercept, standardize === :orthonormal)
  m = CDLeastSquaresLoss(y, d.Z; intercept)
  g = GroupNormL2(one(T), d.zgroups, w[d.fitted])
  m, g, AtomIterate(T, size(d.Z, 2), d.zgroups), d, gs, w
end

_groupIntercept(X, y, β, intercept::Bool) = intercept ? mean(y) - dot(_column_means(X), β) : zero(eltype(y))

"""
    GroupLassoSolution

Result returned by [`groupLasso`](@ref).

# Fields
- `x::SparseVector`: the coefficients `β`, on the scale of the data
- `intercept`: the unpenalized intercept `β₀`, zero unless `intercept = true`
- `λ`: the penalty level
- `groups::Vector{Vector{Int}}`, `groupweights::Vector`: the groups and their
  weights `wₖ`
- `activegroups::Vector{Int}`: the groups with nonzero coefficients
- `iterations::Int`, `converged::Bool`: the passes of coordinate descent and
  whether it met its stopping rule
- `hasintercept::Bool`, `nobs::Int`: whether the solver fitted an intercept, and
  the number of observations

[`coef`](@ref), [`predict`](@ref), [`nobs`](@ref) and [`dof`](@ref) apply to it.
"""
struct GroupLassoSolution{T}
  x::SparseVector{T, Int}
  intercept::T
  λ::T
  groups::Vector{Vector{Int}}
  groupweights::Vector{T}
  activegroups::Vector{Int}
  iterations::Int
  converged::Bool
  hasintercept::Bool
  nobs::Int
end

"""
    groupLasso(X, y, λ, groups, options = CDOptions(); intercept = true, standardize = :orthonormal,
               groupweights = sqrt.(length.(groups)))

The group lasso

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ₖ wₖ ‖X̃_Gₖ βGₖ‖₂ / √n

for disjoint groups `Gₖ` that cover the columns of `X` (give a column its own
group to treat it alone, and a zero weight to leave a group unpenalized), with
an unpenalized intercept `β₀` that is zero unless `intercept` is true, and
`X̃_G` the columns of `G`, centered with an intercept. The weights default to
`wₖ = √|Gₖ|`.

With `standardize = :orthonormal`, the default, the penalty is on the fitted
values of each group, as above, which makes the estimate invariant to a change
of basis within a group (Simon and Tibshirani, 2012; R's grpreg). It is fitted
on orthonormalized groups: each `X̃_G` is factored by a pivoted QR as `Z_G R̃`,
with `Z_G'Z_G / n = I`, the group lasso is solved in the coefficients `θ_G =
R̃ β_G` by block coordinate descent, whose steps are then exact, and mapped
back. The columns of a group that are linear combinations of its other columns
get zero coefficients. With `standardize = :none`, the penalty is
`λ ∑ₖ wₖ ‖βGₖ‖₂` on the columns as given.

Coordinate descent runs with `options`; with `options.warmStart` false the
solution is computed along `options.numSteps` levels from λmax. Returns a
[`GroupLassoSolution`](@ref). See also [`GroupLassoPath`](@ref).
"""
function groupLasso(X::AbstractMatrix{T}, y::AbstractVector{T}, λ::T, groups::AbstractVector,
                    options::CDOptions = CDOptions(); intercept::Bool = true, standardize::Symbol = :orthonormal,
                    groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  _check_finite(X, "X"); _check_finite(y, "y")
  p = size(X, 2)
  m, g1, x, d, gs, w = _groupProblem(X, y, groups, intercept, standardize, groupweights)
  r = _solve!(x, m, GroupNormL2(λ, g1.groups, g1.w), options)
  β = _coefficients(d, Vector(x), p)
  GroupLassoSolution{T}(β, _groupIntercept(X, y, β, intercept), λ, gs, w, _activegroups(gs, β), r.passes,
                        r.converged, intercept, size(X, 1))
end

function groupLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real, groups::AbstractVector,
                    options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  groupLasso(Xf, yf, T(λ), groups, options; kwargs...)
end

"""
    GroupLassoPath(X, y, groups, options = CDOptions(); nλ = 100, λminratio, kwargs...)
    GroupLassoPath(X, y, λpath, groups, options = CDOptions(); intercept = true, standardize = :orthonormal,
                   groupweights = sqrt.(length.(groups)), screening = true, max_groups = Inf)

Solutions `β0path[k]` and `βpath[k]` of the group lasso of [`groupLasso`](@ref)
along a decreasing sequence `λpath`, each one warm started from the previous
solution, by ProximalSolvers' `solve_path`.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`.

With `screening`, each λ is solved first over the groups kept by the sequential
strong rule, and the optimality conditions of the others are checked afterwards
(`kktChecks[k]` times at `λpath[k]`). The path stops early after the first
solution with more than `max_groups` nonzero groups; the returned `λpath` then
contains only the values that were solved for. `activegroups[k]` lists the
nonzero groups at `λpath[k]`, and `iterations[k]` and `converged[k]` are the
passes of coordinate descent there and whether it met its stopping rule.
`coef`, `predict` and `nobs` apply to the path, as for [`LassoPath`](@ref), and
`fit(GroupLassoPath, X, y, ...)` is `GroupLassoPath(X, y, ...)`.
"""
struct GroupLassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseVector{T, Int}}
  β0path::Vector{T}
  activegroups::Vector{Vector{Int}}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kktChecks::Vector{Int}
  intercept::Bool
  groups::Vector{Vector{Int}}
  groupweights::Vector{T}
  nobs::Int
end

function GroupLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, λpath::Vector{T}, groups::AbstractVector,
                        options::CDOptions = CDOptions(); intercept::Bool = true, standardize::Symbol = :orthonormal,
                        groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing, screening::Bool = true,
                        max_groups = Inf) where {T<:AbstractFloat}
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  _check_finite(X, "X"); _check_finite(y, "y")
  p = size(X, 2)
  m, g1, x, d, gs, w = _groupProblem(X, y, groups, intercept, standardize, groupweights)
  path = solve_path(m, g1, λpath, _solver(options); x, screening, maxnnz = max_groups)
  βpath = [_coefficients(d, θ, p) for θ in path.solutions]
  GroupLassoPath{T}(path.λ, βpath, [_groupIntercept(X, y, β, intercept) for β in βpath],
                    [_activegroups(gs, β) for β in βpath], path.passes, path.converged, path.kkt_checks,
                    intercept, gs, w, size(X, 1))
end

function GroupLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, groups::AbstractVector,
                        options::CDOptions = CDOptions(); nλ::Integer = 100,
                        λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4), intercept::Bool = true,
                        standardize::Symbol = :orthonormal,
                        groupweights::Union{Nothing, AbstractVector{<:Real}} = nothing, kwargs...) where {T<:AbstractFloat}
  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  _check_finite(X, "X"); _check_finite(y, "y")
  m, g1, x, _, _, _ = _groupProblem(X, y, groups, intercept, standardize, groupweights)
  λmax = lambda_max(m, g1, x) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  GroupLassoPath(X, y, λpath, groups, options; intercept, standardize, groupweights, kwargs...)
end

function GroupLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λpath::AbstractVector{<:Real},
                        groups::AbstractVector, options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  GroupLassoPath(Xf, yf, convert(Vector{T}, λpath), groups, options; kwargs...)
end

function GroupLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, groups::AbstractVector,
                        options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  GroupLassoPath(Xf, yf, groups, options; kwargs...)
end
