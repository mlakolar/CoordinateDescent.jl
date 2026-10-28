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
computed from sufficient statistics (`scaled_lasso!` with a
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
StatsAPI.coef(path::LassoPath{T}) where {T} = _coefmatrix(T, path.βpath, path.β0path, path.intercept)
StatsAPI.coef(path::LassoPath{T}, λ::Real) where {T} = _coefat(T, path.λpath, _coefk(path), λ)

# the coefficients along a path, a column per level, with the intercepts in the
# first row when the path has them
function _coefmatrix(::Type{T}, βpath, β0path, intercept::Bool) where {T}
  p = isempty(βpath) ? 0 : length(βpath[1])
  B = isempty(βpath) ? spzeros(T, p, 0) : reduce(hcat, βpath)
  intercept || return B
  dropzeros!(vcat(sparse(reshape(β0path, 1, :)), B))
end

# the coefficients at the k-th level of a path with βpath, β0path and intercept
_coefk(path) = k -> path.intercept ? _prepend(path.β0path[k], path.βpath[k]) : path.βpath[k]

# the coefficients at λ, coefk(k) at the levels λs, interpolated linearly in λ
# between neighboring levels
function _coefat(::Type{T}, λs, coefk, λ::Real) where {T}
  isempty(λs) && throw(ArgumentError("the path is empty."))
  λs[end] <= λ <= λs[1] || throw(ArgumentError("λ = $λ is outside the range [$(λs[end]), $(λs[1])] of the path."))
  k = findlast(>=(λ), λs)::Int        # λs[end] ≤ λ, so k exists
  (λs[k] == λ || k == length(λs)) && return coefk(k)
  t = T((λs[k] - λ) / (λs[k] - λs[k+1]))
  dropzeros!((1 - t) * coefk(k) + t * coefk(k + 1))
end

"""
    predict(path::LassoPath, X)

The fitted values `β₀ .+ X * β` along the path: a matrix with one column per
value of `path.λpath`.
"""
StatsAPI.predict(path::LassoPath{T}, X::AbstractMatrix{<:Real}) where {T} = _predictpath(T, path.βpath, path.β0path, X)

# the fitted values β₀ .+ X * β at each level of a path, a column per level
function _predictpath(::Type{T}, βpath, β0path, X::AbstractMatrix) where {T}
  p = isempty(βpath) ? size(X, 2) : length(βpath[1])
  size(X, 2) == p || throw(DimensionMismatch("X has $(size(X, 2)) columns, but the path has $p coefficients."))
  B = isempty(βpath) ? spzeros(T, p, 0) : reduce(hcat, βpath)
  Ŷ = Matrix(X * B)
  Ŷ .+= transpose(β0path)
end

function _checkcolumns(X::AbstractMatrix, p::Integer)
  size(X, 2) == p || throw(DimensionMismatch("X has $(size(X, 2)) columns, but the model has $p coefficients."))
  nothing
end

"""
    fit(LassoPath, X, y, args...; kwargs...)

The same as [`LassoPath`](@ref)`(X, y, args...; kwargs...)`.
"""
StatsAPI.fit(::Type{LassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  LassoPath(X, y, args...; kwargs...)

"""
    fit(CVLassoPath, X, y, args...; kwargs...)
    fit(GLMLassoPath, X, y, args...; kwargs...)
    fit(CVGLMLassoPath, X, y, args...; kwargs...)
    fit(GroupLassoPath, X, y, args...; kwargs...)
    fit(SparseGroupLassoPath, X, y, args...; kwargs...)
    fit(MultiLassoPath, X, Y, args...; kwargs...)
    fit(CVMultiLassoPath, X, Y, args...; kwargs...)
    fit(MultitaskLassoPath, Xs, ys, args...; kwargs...)
    fit(CVMultitaskLassoPath, Xs, ys, args...; kwargs...)

The same as calling the constructor, for example
[`GLMLassoPath`](@ref)`(X, y, args...; kwargs...)`.
"""
StatsAPI.fit(::Type{CVLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  CVLassoPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{GLMLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  GLMLassoPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{CVGLMLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  CVGLMLassoPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{GroupLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  GroupLassoPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{SparseGroupLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  SparseGroupLassoPath(X, y, args...; kwargs...)
StatsAPI.fit(::Type{MultiLassoPath}, X::AbstractMatrix, Y::AbstractMatrix, args...; kwargs...) =
  MultiLassoPath(X, Y, args...; kwargs...)
StatsAPI.fit(::Type{CVMultiLassoPath}, X::AbstractMatrix, Y::AbstractMatrix, args...; kwargs...) =
  CVMultiLassoPath(X, Y, args...; kwargs...)
StatsAPI.fit(::Type{MultitaskLassoPath}, Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
             args...; kwargs...) = MultitaskLassoPath(Xs, ys, args...; kwargs...)
StatsAPI.fit(::Type{CVMultitaskLassoPath}, Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector},
             args...; kwargs...) = CVMultitaskLassoPath(Xs, ys, args...; kwargs...)

######################################################################
#
#   Generalized linear models
#
######################################################################

# the mean of the response at the linear predictor η: the inverse of the
# canonical link of the logistic and Poisson losses, η itself for least squares
# and the Huber loss
_inverselink(::ProximalBase.Logistic, η::Real) = η >= 0 ? inv(1 + exp(-η)) : exp(η) / (1 + exp(η))
_inverselink(::ProximalBase.Poisson, η::Real) = exp(η)
_inverselink(::Union{ProximalBase.Squared, ProximalBase.Huber}, η::Real) = η
_inverselink(loss, ::Real) =
  throw(ArgumentError("the mean of the response is not known for the loss $(typeof(loss)); use type = :link."))

function _glmscale(loss, η::AbstractArray, type::Symbol)
  type === :link && return η
  type === :response && return _inverselink.(Ref(loss), η)
  throw(ArgumentError("type must be :response or :link, but it is :$type."))
end

function _addoffset!(η::AbstractArray, offset)
  offset === nothing && return η
  length(offset) == size(η, 1) ||
    throw(DimensionMismatch("offset has length $(length(offset)), but X has $(size(η, 1)) rows."))
  η .+= offset
end

"""
    coef(sol::GLMLassoSolution)
    predict(sol::GLMLassoSolution, X; type = :response, offset = nothing)
    fitted(sol::GLMLassoSolution)

The coefficients `[β₀; β]` (`β` without an intercept), as a `SparseVector`. The
predictions at the rows of `X`, on the scale of the response (`type =
:response`, the probabilities of logistic regression and the means of Poisson
regression) or of the linear predictor `β₀ + Xβ + offset` (`type = :link`).
`fitted` is the response scale at the data of the fit. `nobs`, `dof` (nonzero
coefficients, plus one for the intercept), `deviance` and `nulldeviance` apply
too.
"""
function StatsAPI.coef(sol::GLMLassoSolution)
  β = _sparsevec(sol.x)
  sol.hasintercept ? _prepend(sol.intercept, β) : β
end

function StatsAPI.predict(sol::GLMLassoSolution, X::AbstractMatrix{<:Real}; type::Symbol = :response, offset = nothing)
  _checkcolumns(X, length(sol.x))
  η = Vector(X * _sparsevec(sol.x)) .+ sol.intercept
  _glmscale(sol.loss, _addoffset!(η, offset), type)
end

StatsAPI.fitted(sol::GLMLassoSolution) = _glmscale(sol.loss, sol.linearpredictor, :response)
StatsAPI.nobs(sol::GLMLassoSolution) = length(sol.linearpredictor)
StatsAPI.dof(sol::GLMLassoSolution) = nnz(_sparsevec(sol.x)) + sol.hasintercept
StatsAPI.deviance(sol::GLMLassoSolution) = sol.deviance
StatsAPI.nulldeviance(sol::GLMLassoSolution) = sol.nulldeviance

"""
    coef(path::GLMLassoPath)
    coef(path::GLMLassoPath, λ)
    predict(path::GLMLassoPath, X; type = :response, offset = nothing)

The coefficients along the path (a sparse matrix, a column per level, the
intercepts in the first row) or at `λ` (interpolated linearly in `λ`), as for
[`LassoPath`](@ref), and the predictions at the rows of `X` at every level, on
the scale of the response or of the linear predictor (see `predict` for a
[`GLMLassoSolution`](@ref)).
"""
StatsAPI.coef(path::GLMLassoPath{T}) where {T} = _coefmatrix(T, path.βpath, path.β0path, path.intercept)
StatsAPI.coef(path::GLMLassoPath{T}, λ::Real) where {T} = _coefat(T, path.λpath, _coefk(path), λ)

function StatsAPI.predict(path::GLMLassoPath{T}, X::AbstractMatrix{<:Real}; type::Symbol = :response,
                          offset = nothing) where {T}
  _glmscale(path.loss, _addoffset!(_predictpath(T, path.βpath, path.β0path, X), offset), type)
end

StatsAPI.nobs(path::GLMLassoPath) = path.nobs
StatsAPI.nulldeviance(path::GLMLassoPath) = path.nulldeviance

######################################################################
#
#   The group lasso
#
######################################################################

"""
    coef(sol::GroupLassoSolution)
    predict(sol::GroupLassoSolution, X)

The coefficients `[β₀; β]` (`β` without an intercept) as a `SparseVector`, and
the fitted values `β₀ .+ X * β` at the rows of `X`. `nobs` and `dof` (the
nonzero coefficients, plus one for the intercept) apply too.
"""
StatsAPI.coef(sol::GroupLassoSolution) = sol.hasintercept ? _prepend(sol.intercept, sol.x) : sol.x

function StatsAPI.predict(sol::GroupLassoSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, length(sol.x))
  Vector(X * sol.x) .+ sol.intercept
end

StatsAPI.nobs(sol::GroupLassoSolution) = sol.nobs
StatsAPI.dof(sol::GroupLassoSolution) = nnz(sol.x) + sol.hasintercept

"""
    coef(path::GroupLassoPath)
    coef(path::GroupLassoPath, λ)
    predict(path::GroupLassoPath, X)

As for [`LassoPath`](@ref).
"""
StatsAPI.coef(path::GroupLassoPath{T}) where {T} = _coefmatrix(T, path.βpath, path.β0path, path.intercept)
StatsAPI.coef(path::GroupLassoPath{T}, λ::Real) where {T} = _coefat(T, path.λpath, _coefk(path), λ)
StatsAPI.predict(path::GroupLassoPath{T}, X::AbstractMatrix{<:Real}) where {T} =
  _predictpath(T, path.βpath, path.β0path, X)
StatsAPI.nobs(path::GroupLassoPath) = path.nobs

"""
    coef(sol::SparseGroupLassoSolution)
    predict(sol::SparseGroupLassoSolution, X)
    coef(path::SparseGroupLassoPath)
    coef(path::SparseGroupLassoPath, λ)
    predict(path::SparseGroupLassoPath, X)

As for the group lasso: the coefficients `[β₀; β]` (`β` without an intercept)
and the fitted values `β₀ .+ X * β`, along the path as for
[`LassoPath`](@ref). `nobs` and `dof` (the nonzero coefficients, plus one for
the intercept) apply too.
"""
StatsAPI.coef(sol::SparseGroupLassoSolution) = sol.hasintercept ? _prepend(sol.intercept, sol.x) : sol.x

function StatsAPI.predict(sol::SparseGroupLassoSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, length(sol.x))
  Vector(X * sol.x) .+ sol.intercept
end

StatsAPI.nobs(sol::SparseGroupLassoSolution) = sol.nobs
StatsAPI.dof(sol::SparseGroupLassoSolution) = nnz(sol.x) + sol.hasintercept
StatsAPI.coef(path::SparseGroupLassoPath{T}) where {T} = _coefmatrix(T, path.βpath, path.β0path, path.intercept)
StatsAPI.coef(path::SparseGroupLassoPath{T}, λ::Real) where {T} = _coefat(T, path.λpath, _coefk(path), λ)
StatsAPI.predict(path::SparseGroupLassoPath{T}, X::AbstractMatrix{<:Real}) where {T} =
  _predictpath(T, path.βpath, path.β0path, X)
StatsAPI.nobs(path::SparseGroupLassoPath) = path.nobs

######################################################################
#
#   The multi-response lasso
#
######################################################################

# [β₀'; B], the intercepts of the responses in the first row
_prependrow(β0::AbstractVector, B::SparseMatrixCSC) = dropzeros!(vcat(sparse(reshape(β0, 1, :)), B))

"""
    coef(sol::MultiLassoSolution)
    predict(sol::MultiLassoSolution, X)

The `p × q` coefficients `B` as a sparse matrix, with the intercepts of the
responses in a first row when the solver fitted them, and the `n × q` fitted
values `1β₀' + XB` at the rows of `X`. `nobs` and `dof` (the nonzero
coefficients, plus one intercept per response) apply too.
"""
StatsAPI.coef(sol::MultiLassoSolution) = sol.hasintercept ? _prependrow(sol.intercept, sol.B) : sol.B

function StatsAPI.predict(sol::MultiLassoSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, size(sol.B, 1))
  Matrix(X * sol.B) .+ transpose(sol.intercept)
end

StatsAPI.nobs(sol::MultiLassoSolution) = sol.nobs
StatsAPI.dof(sol::MultiLassoSolution) = nnz(sol.B) + sol.hasintercept * size(sol.B, 2)

"""
    coef(path::MultiLassoPath)
    coef(path::MultiLassoPath, λ)
    predict(path::MultiLassoPath, X)

The coefficient matrices along the path, one per level (with the intercepts in
the first row when the path has them), or the matrix at `λ`, interpolated
linearly in `λ` between neighboring levels; and the matrices of fitted values
at the rows of `X`, one per level.
"""
StatsAPI.coef(path::MultiLassoPath) = [_multicoef(path, k) for k in eachindex(path.λpath)]
StatsAPI.coef(path::MultiLassoPath{T}, λ::Real) where {T} = _coefat(T, path.λpath, k -> _multicoef(path, k), λ)
_multicoef(path::MultiLassoPath, k::Int) = path.intercept ? _prependrow(path.β0path[k], path.Bpath[k]) : path.Bpath[k]

function StatsAPI.predict(path::MultiLassoPath, X::AbstractMatrix{<:Real})
  isempty(path.Bpath) || _checkcolumns(X, size(path.Bpath[1], 1))
  [Matrix(X * path.Bpath[k]) .+ transpose(path.β0path[k]) for k in eachindex(path.λpath)]
end

StatsAPI.nobs(path::MultiLassoPath) = path.nobs

"""
    coef(cv::CVMultiLassoPath; λ = :λ1se)
    predict(cv::CVMultiLassoPath, X; λ = :λ1se)

The coefficients (with intercepts in the first row when fitted), and the
matrix of predictions at the rows of X, at `cv.λ1se`, at `cv.λmin` with
`λ = :λmin`, or at a level within the path.
"""
StatsAPI.coef(cv::CVMultiLassoPath; λ::Union{Symbol, Real} = :λ1se) = coef(cv.path, _cvlevel(cv, λ))

function StatsAPI.predict(cv::CVMultiLassoPath, X::AbstractMatrix{<:Real}; λ::Union{Symbol, Real} = :λ1se)
  b = coef(cv; λ)
  _checkcolumns(X, size(first(cv.path.Bpath), 1))
  cv.path.intercept ? Matrix(X * b[2:end, :]) .+ b[1, :]' : Matrix(X * b)
end

######################################################################
#
#   The multitask lasso
#
######################################################################

# the fitted values β₀ₜ .+ X_t * b_t at the designs Xs of the tasks, the columns
# of B
function _taskpredict(Xs::AbstractVector{<:AbstractMatrix}, B::SparseMatrixCSC, β0::AbstractVector)
  length(Xs) == size(B, 2) ||
    throw(DimensionMismatch("there are $(length(Xs)) designs, but the model has $(size(B, 2)) tasks."))
  foreach(X -> _checkcolumns(X, size(B, 1)), Xs)
  [Vector(Xs[t] * B[:, t]) .+ β0[t] for t in eachindex(Xs)]
end

"""
    coef(sol::MultitaskLassoSolution)
    predict(sol::MultitaskLassoSolution, Xs)

The `p × q` coefficients `B` as a sparse matrix, a column per task, with the
intercepts of the tasks in a first row when the solver fitted them, and the
fitted values `β₀ₜ .+ X_t * b_t` at the rows of the designs `Xs[t]` of the
tasks, a vector per task. `nobs` (the observations of all the tasks) and `dof`
(the nonzero coefficients, plus one intercept per task) apply too.
"""
StatsAPI.coef(sol::MultitaskLassoSolution) = sol.hasintercept ? _prependrow(sol.intercept, sol.B) : sol.B
StatsAPI.predict(sol::MultitaskLassoSolution, Xs::AbstractVector{<:AbstractMatrix}) =
  _taskpredict(Xs, sol.B, sol.intercept)
StatsAPI.nobs(sol::MultitaskLassoSolution) = sol.nobs
StatsAPI.dof(sol::MultitaskLassoSolution) = nnz(sol.B) + sol.hasintercept * size(sol.B, 2)

"""
    coef(path::MultitaskLassoPath)
    coef(path::MultitaskLassoPath, λ)
    predict(path::MultitaskLassoPath, Xs)

The coefficient matrices along the path, one per level (with the intercepts in
the first row when the path has them), or the matrix at `λ`, interpolated
linearly in `λ` between neighboring levels; and the fitted values at the rows
of the designs `Xs[t]` of the tasks, an `nₜ × L` matrix per task with a column
per level.
"""
StatsAPI.coef(path::MultitaskLassoPath) = [_multitaskcoef(path, k) for k in eachindex(path.λpath)]
StatsAPI.coef(path::MultitaskLassoPath{T}, λ::Real) where {T} =
  _coefat(T, path.λpath, k -> _multitaskcoef(path, k), λ)
_multitaskcoef(path::MultitaskLassoPath, k::Int) =
  path.intercept ? _prependrow(path.β0path[k], path.Bpath[k]) : path.Bpath[k]

function StatsAPI.predict(path::MultitaskLassoPath{T}, Xs::AbstractVector{<:AbstractMatrix}) where {T}
  q = length(path.task_nobs)
  length(Xs) == q || throw(DimensionMismatch("there are $(length(Xs)) designs, but the path has $q tasks."))
  [_predictpath(T, [B[:, t] for B in path.Bpath], T[β0[t] for β0 in path.β0path], Xs[t]) for t in 1:q]
end

StatsAPI.nobs(path::MultitaskLassoPath) = path.nobs

"""
    coef(cv::CVMultitaskLassoPath; λ = :λ1se)
    predict(cv::CVMultitaskLassoPath, Xs; λ = :λ1se)

The coefficients (with the intercepts in the first row when the path has them),
and the fitted values at the rows of the designs `Xs[t]` of the tasks, a vector
per task, of the path on all the data at `cv.λ1se`, at `cv.λmin` with
`λ = :λmin`, or at a level `λ` within the path.
"""
StatsAPI.coef(cv::CVMultitaskLassoPath; λ::Union{Symbol, Real} = :λ1se) = coef(cv.path, _cvlevel(cv, λ))

function StatsAPI.predict(cv::CVMultitaskLassoPath{T}, Xs::AbstractVector{<:AbstractMatrix};
                          λ::Union{Symbol, Real} = :λ1se) where {T}
  C = coef(cv; λ)
  cv.path.intercept || return _taskpredict(Xs, C, zeros(T, size(C, 2)))
  _taskpredict(Xs, C[2:end, :], Vector(C[1, :]))
end

######################################################################
#
#   Inference after selection
#
######################################################################

"""
    coef(sol::RLassoSolution)
    predict(sol::RLassoSolution, X)

The coefficients `[β₀; β]` (`β` without an intercept) as a `SparseVector`, and
the fitted values `β₀ .+ X * β` at the rows of `X`. `residuals`, `nobs`, `dof`
(the selected columns, plus one for the intercept) and `deviance` (the
residual sum of squares) apply too.
"""
StatsAPI.coef(sol::RLassoSolution) = sol.hasintercept ? _prepend(sol.intercept, sol.x) : sol.x

function StatsAPI.predict(sol::RLassoSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, length(sol.x))
  Vector(X * sol.x) .+ sol.intercept
end

StatsAPI.residuals(sol::RLassoSolution) = sol.residuals
StatsAPI.nobs(sol::RLassoSolution) = length(sol.residuals)
StatsAPI.dof(sol::RLassoSolution) = length(sol.selected) + sol.hasintercept
StatsAPI.deviance(sol::RLassoSolution) = sum(abs2, sol.residuals)

"""
    coef(r::PostLassoResult)
    stderror(r::PostLassoResult)
    vcov(r::PostLassoResult)
    confint(r::PostLassoResult; level = 0.95)

The least-squares coefficients `[β₀; β]` (`β` without an intercept) and their
HC1 standard errors as `SparseVector`s, zero off the selected columns, and
their covariance as a sparse matrix. `confint` gives normal intervals
`coef ± z stderror` at `level`, as a matrix with a row per coefficient; those
off the selected columns are `[0, 0]`, since the coefficients were not
estimated. `predict`, `residuals`, `nobs`, `dof` (the coefficients estimated)
and `deviance` (the residual sum of squares) apply too.
"""
StatsAPI.coef(r::PostLassoResult) = r.hasintercept ? _prepend(r.intercept, r.x) : r.x
StatsAPI.stderror(r::PostLassoResult) = r.hasintercept ? _prepend(r.se_intercept, r.se) : r.se

function StatsAPI.vcov(r::PostLassoResult{T}) where {T}
  idx = r.hasintercept ? [1; r.selected .+ 1] : r.selected
  k = length(r.x) + r.hasintercept
  I, J = repeat(idx, outer = length(idx)), repeat(idx, inner = length(idx))
  sparse(I, J, vec(r.vcov), k, k)
end

function StatsAPI.confint(r::PostLassoResult{T}; level::Real = 0.95) where {T}
  0 < level < 1 || throw(ArgumentError("level must be in (0, 1), but it is $level."))
  z = _qnorm(1 - (1 - T(level)) / 2)
  b, se = Vector(coef(r)), Vector(stderror(r))
  [b .- z .* se b .+ z .* se]
end

function StatsAPI.predict(r::PostLassoResult, X::AbstractMatrix{<:Real})
  _checkcolumns(X, length(r.x))
  Vector(X * r.x) .+ r.intercept
end

StatsAPI.residuals(r::PostLassoResult) = r.residuals
StatsAPI.nobs(r::PostLassoResult) = length(r.residuals)
StatsAPI.dof(r::PostLassoResult) = length(r.selected) + r.hasintercept
StatsAPI.deviance(r::PostLassoResult) = sum(abs2, r.residuals)

"""
    coef(te::TreatmentEffect)
    stderror(te::TreatmentEffect)
    vcov(te::TreatmentEffect)
    confint(te::TreatmentEffect; level = 0.95)

The estimate of the treatment effect as a model with one coefficient: a vector
of length one, its standard error, a 1 × 1 covariance using `te.vcov_type`, and
the normal confidence interval at `level` as a 1 × 2 matrix (`te.ci` holds the one at the
level of [`double_selection`](@ref)). `nobs` is the number of observations.
"""
StatsAPI.coef(te::TreatmentEffect) = [te.estimate]
StatsAPI.stderror(te::TreatmentEffect) = [te.se]
StatsAPI.vcov(te::TreatmentEffect) = fill(te.se^2, 1, 1)
StatsAPI.nobs(te::TreatmentEffect) = te.n

function StatsAPI.confint(te::TreatmentEffect{T}; level::Real = 0.95) where {T}
  0 < level < 1 || throw(ArgumentError("level must be in (0, 1), but it is $level."))
  z = _qnorm(1 - (1 - T(level)) / 2)
  [te.estimate - z * te.se te.estimate + z * te.se]
end
