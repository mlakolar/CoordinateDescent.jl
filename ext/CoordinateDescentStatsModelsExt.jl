module CoordinateDescentStatsModelsExt

using CoordinateDescent
using CoordinateDescent: FormulaFit
using StatsModels
using StatsModels: FormulaTerm, InterceptTerm, Tables, apply_schema, modelcols, schema, termvars, width
using StatsAPI: StatsAPI, StatisticalModel

# The data of a formula fit: the rows without a missing value in the variables
# of the formula, the weights or the offset, and the formula with its schema
# applied (with an implicit intercept, as in GLM.jl).
function _formuladata(f::FormulaTerm, data, contrasts, extra)
  Tables.istable(data) || throw(ArgumentError("the data must be a table, such as a DataFrame or a NamedTuple of vectors."))
  cols = Tables.columntable(data)
  vars = unique([termvars(f); [s for s in extra if s isa Symbol]])
  for v in vars
    haskey(cols, v) || throw(ArgumentError("the data have no column $v."))
  end
  keep = trues(Tables.rowcount(cols))
  for v in vars
    keep .&= .!ismissing.(cols[v])
  end
  any(keep) || throw(ArgumentError("every row has a missing value in a variable of the model."))
  sub = NamedTuple{Tuple(vars)}(Tuple(_nonmissing(cols[v][keep]) for v in vars))
  fs = apply_schema(f, schema(f, sub, contrasts), StatisticalModel)
  fs, sub
end

_nonmissing(v::AbstractVector) = Missing <: eltype(v) ? convert(Vector{nonmissingtype(eltype(v))}, v) : v

# a weights or offset keyword: nothing, a vector, or the name of a column
_column(::Nothing, sub) = nothing
_column(v::AbstractVector, sub) = v
_column(s::Symbol, sub) = sub[s]

# The model matrix without the intercept column, whether the formula has an
# intercept, the columns kept, the groups of columns of the terms (for the
# group lasso), and the names of the coefficients in the order of coef: the
# intercept first.
function _design(fs::FormulaTerm, sub)
  y, Xall = modelcols(fs, sub)
  terms = fs.rhs.terms                           # the schema makes the right side a MatrixTerm
  names = collect(String, StatsModels.coefnames(fs.rhs))
  hasintercept = any(t -> t isa InterceptTerm{true}, terms)
  columns, groups = Int[], UnitRange{Int}[]
  j = 0
  for t in terms
    w = width(t)
    if !(t isa InterceptTerm)
      push!(groups, length(columns)+1:length(columns)+w)
      append!(columns, j+1:j+w)
    end
    j += w
  end
  X = Xall[:, columns]
  coefnames = hasintercept ? ["(Intercept)"; names[columns]] : names[columns]
  y, X, hasintercept, columns, groups, coefnames
end

function _checkkeywords(kwargs)
  haskey(kwargs, :intercept) &&
    throw(ArgumentError("the formula sets the intercept: write `0 +` in it to fit without one."))
  nothing
end

# Fits fitter(X, y, hasintercept, groups, w, o) on the data of the formula and
# wraps the result.
function _formulafit(fitter, f::FormulaTerm, data; contrasts = Dict{Symbol, Any}(), weights = nothing,
                     offset = nothing)
  fs, sub = _formuladata(f, data, contrasts, (weights, offset))
  y, X, hasintercept, columns, groups, coefnames = _design(fs, sub)
  y isa AbstractVector || throw(ArgumentError("the formula must have one response."))
  model = fitter(X, y, hasintercept, groups, _column(weights, sub), _column(offset, sub))
  FormulaFit(model, fs, coefnames, StatsModels.coefnames(fs.lhs), columns)
end

# the keywords weights and offset, passed on when they are given
_wo(w, o) = (; (w === nothing ? () : (weights = w,))..., (o === nothing ? () : (offset = o,))...)

for T in (LassoPath, CVLassoPath)
  @eval function StatsAPI.fit(::Type{$T}, f::FormulaTerm, data, args...; contrasts = Dict{Symbol, Any}(),
                              weights = nothing, offset = nothing, kwargs...)
    _checkkeywords(kwargs)
    _formulafit(f, data; contrasts, weights, offset) do X, y, intercept, groups, w, o
      $T(X, y, args...; intercept, _wo(w, o)..., kwargs...)
    end
  end
end

for T in (GLMLassoPath, CVGLMLassoPath)
  @eval function StatsAPI.fit(::Type{$T}, f::FormulaTerm, data, args...; contrasts = Dict{Symbol, Any}(),
                              weights = nothing, offset = nothing, kwargs...)
    _checkkeywords(kwargs)
    _formulafit(f, data; contrasts, weights, offset) do X, y, intercept, groups, w, o
      $T(X, y, args...; intercept, weights = w, offset = o, kwargs...)
    end
  end
end

# the groups of the group lasso: one per term of the formula, unless given;
# they go after λpath and before the options
function StatsAPI.fit(::Type{GroupLassoPath}, f::FormulaTerm, data, args...; contrasts = Dict{Symbol, Any}(),
                      groups = nothing, kwargs...)
  _checkkeywords(kwargs)
  λargs, oargs = filter(a -> !(a isa CDOptions), args), filter(a -> a isa CDOptions, args)
  _formulafit(f, data; contrasts) do X, y, intercept, termgroups, w, o
    GroupLassoPath(X, y, λargs..., something(groups, termgroups), oargs...; intercept, kwargs...)
  end
end

for fn in (:lasso, :sqrt_lasso, :adaptive_lasso)
  @eval function CoordinateDescent.$fn(f::FormulaTerm, data, λ::Real, args...; contrasts = Dict{Symbol, Any}(),
                                       weights = nothing, offset = nothing, kwargs...)
    _checkkeywords(kwargs)
    _formulafit(f, data; contrasts, weights, offset) do X, y, intercept, groups, w, o
      $fn(X, y, λ, args...; intercept, _wo(w, o)..., kwargs...)
    end
  end
end

function CoordinateDescent.glm_lasso(f::FormulaTerm, data, λ::Real, args...; contrasts = Dict{Symbol, Any}(),
                                    weights = nothing, offset = nothing, kwargs...)
  _checkkeywords(kwargs)
  _formulafit(f, data; contrasts, weights, offset) do X, y, intercept, groups, w, o
    glm_lasso(X, y, λ, args...; intercept, weights = w, offset = o, kwargs...)
  end
end

# the sparse group lasso, with the groups of the terms unless given
function StatsAPI.fit(::Type{SparseGroupLassoPath}, f::FormulaTerm, data, args...; contrasts = Dict{Symbol, Any}(),
                      groups = nothing, kwargs...)
  _checkkeywords(kwargs)
  λargs, oargs = filter(a -> !(a isa CDOptions), args), filter(a -> a isa CDOptions, args)
  _formulafit(f, data; contrasts) do X, y, intercept, termgroups, w, o
    SparseGroupLassoPath(X, y, λargs..., something(groups, termgroups), oargs...; intercept, kwargs...)
  end
end

function CoordinateDescent.sparse_group_lasso(f::FormulaTerm, data, λ::Real, args...; contrasts = Dict{Symbol, Any}(),
                                            groups = nothing, kwargs...)
  _checkkeywords(kwargs)
  _formulafit(f, data; contrasts) do X, y, intercept, termgroups, w, o
    sparse_group_lasso(X, y, λ, something(groups, termgroups), args...; intercept, kwargs...)
  end
end

function CoordinateDescent.group_lasso(f::FormulaTerm, data, λ::Real, args...; contrasts = Dict{Symbol, Any}(),
                                      groups = nothing, kwargs...)
  _checkkeywords(kwargs)
  _formulafit(f, data; contrasts) do X, y, intercept, termgroups, w, o
    group_lasso(X, y, λ, something(groups, termgroups), args...; intercept, kwargs...)
  end
end

function CoordinateDescent.rlasso(f::FormulaTerm, data, args...; contrasts = Dict{Symbol, Any}(), kwargs...)
  _checkkeywords(kwargs)
  _formulafit(f, data; contrasts) do X, y, intercept, groups, w, o
    rlasso(X, y, args...; intercept, kwargs...)
  end
end

# predict(m::FormulaFit, data; kwargs...): the predictions of the model at the
# rows of the table data, whose columns the formula builds as at the fit (with
# the levels of the categorical variables of the fit); the keywords go to the
# model's predict. The rows must not have missing values in the variables of
# the model. Documented with FormulaFit.
function StatsAPI.predict(m::FormulaFit, data; kwargs...)
  Tables.istable(data) || throw(ArgumentError("the data must be a table, such as a DataFrame or a NamedTuple of vectors."))
  X = modelcols(m.formula.rhs, Tables.columntable(data))[:, m.columns]
  predict(m.model, X; kwargs...)
end

end
