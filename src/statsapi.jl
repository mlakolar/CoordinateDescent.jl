######################################################################
#
#   The StatsAPI interface of the solutions
#
######################################################################

# [β₀; β] as a SparseVector
function _prepend(β0::T, β::SparseVector{T}) where {T}
  iszero(β0) && return sparsevec(β.nzind .+ 1, β.nzval, length(β) + 1)
  sparsevec([1; β.nzind .+ 1], [β0; β.nzval], length(β) + 1)
end

"""
    coef(sol::LassoSolution)

The coefficients as a `SparseVector`: `[β₀; β]`, the intercept first, when the
solver fitted an intercept, and `β` otherwise, so that they line up with a
model matrix whose first column is the constant.
"""
function StatsAPI.coef(sol::LassoSolution)
  β = _sparsevec(sol.x)
  sol.hasintercept ? _prepend(sol.intercept, β) : β
end

"""
    predict(sol::LassoSolution, X)

The fitted values `β₀ .+ X * β` at the rows of `X`, which has the columns of
the data the solver was given (without a column for the intercept); an offset
of the fit is not included.
"""
function StatsAPI.predict(sol::LassoSolution, X::AbstractMatrix{<:Real})
  size(X, 2) == length(sol.x) ||
    throw(DimensionMismatch("X has $(size(X, 2)) columns, but the solution has $(length(sol.x)) coefficients."))
  ŷ = Vector(X * _sparsevec(sol.x))
  ŷ .+= sol.intercept
end

function _residuals(sol::LassoSolution)
  isempty(sol.residuals) &&
    throw(ArgumentError("the solution was computed from sufficient statistics, so it has no residuals."))
  sol.residuals
end

"""
    residuals(sol::LassoSolution)

The residuals `y - β₀ - X * β` of the data the solver was given. A solution
computed from sufficient statistics (`scaledLasso!` with a
[`CDSqrtQuadraticLoss`](@ref)) has none, and an error is thrown; so it is for
[`nobs`](@ref) and [`deviance`](@ref).
"""
StatsAPI.residuals(sol::LassoSolution) = _residuals(sol)

"""
    nobs(sol::LassoSolution)
    nobs(path::LassoPath)

The number of observations the solver was given.
"""
StatsAPI.nobs(sol::LassoSolution) = length(_residuals(sol))
StatsAPI.nobs(path::LassoPath) = path.nobs

"""
    dof(sol::LassoSolution)

The number of nonzero coefficients, plus one for an intercept: an unbiased
estimate of the degrees of freedom of the lasso (Zou, Hastie and Tibshirani,
*On the "degrees of freedom" of the lasso*, 2007). For the elastic net, MCP and
SCAD it is the count that glmnet and ncvreg report, not an unbiased estimate.
"""
StatsAPI.dof(sol::LassoSolution) = nnz(_sparsevec(sol.x)) + sol.hasintercept

"""
    deviance(sol::LassoSolution)

The residual sum of squares `‖y - β₀ - X * β‖²`, weighted, `∑ᵢ wᵢ rᵢ²`, for a fit
with observation weights.
"""
function StatsAPI.deviance(sol::LassoSolution)
  r = _residuals(sol)
  sol.weights === nothing ? sum(abs2, r) : sum(i -> sol.weights[i] * r[i]^2, eachindex(r))
end

"""
    coef(path::LassoPath)
    coef(path::LassoPath, λ)

The coefficients along the path, as a sparse matrix with one column per value
of `path.λpath`, whose first row holds the intercepts when the path has an
intercept. With `λ`, the coefficients at `λ` as a `SparseVector`, interpolated
linearly in `λ` between the two neighboring values of `λpath` (as glmnet's
`coef(fit, s = λ)`), and exact at the values of `λpath`; `λ` must be within
the range of `λpath`.
"""
function StatsAPI.coef(path::LassoPath{T}) where {T}
  p = isempty(path.βpath) ? 0 : length(path.βpath[1])
  B = isempty(path.βpath) ? spzeros(T, p, 0) : reduce(hcat, path.βpath)
  path.intercept || return B
  dropzeros!(vcat(sparse(reshape(path.β0path, 1, :)), B))
end

function StatsAPI.coef(path::LassoPath{T}, λ::Real) where {T}
  λs = path.λpath
  isempty(λs) && throw(ArgumentError("the path is empty."))
  λs[end] <= λ <= λs[1] || throw(ArgumentError("λ = $λ is outside the range [$(λs[end]), $(λs[1])] of the path."))
  k = findlast(>=(λ), λs)::Int        # λs[end] ≤ λ, so k exists
  coefk(j) = path.intercept ? _prepend(path.β0path[j], path.βpath[j]) : path.βpath[j]
  (λs[k] == λ || k == length(λs)) && return coefk(k)
  t = T((λs[k] - λ) / (λs[k] - λs[k+1]))
  dropzeros!((1 - t) * coefk(k) + t * coefk(k + 1))
end

"""
    predict(path::LassoPath, X)

The fitted values `β₀ .+ X * β` along the path: a matrix with one column per
value of `path.λpath`.
"""
function StatsAPI.predict(path::LassoPath{T}, X::AbstractMatrix{<:Real}) where {T}
  p = isempty(path.βpath) ? size(X, 2) : length(path.βpath[1])
  size(X, 2) == p || throw(DimensionMismatch("X has $(size(X, 2)) columns, but the path has $p coefficients."))
  B = isempty(path.βpath) ? spzeros(T, p, 0) : reduce(hcat, path.βpath)
  Ŷ = Matrix(X * B)
  Ŷ .+= transpose(path.β0path)
end

StatsAPI.fit(::Type{LassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  LassoPath(X, y, args...; kwargs...)
