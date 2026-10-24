# Compatibility with the camelCase API of version 0.7.
Base.@deprecate sqrtLasso(args...; kwargs...) sqrt_lasso(args...; kwargs...)
Base.@deprecate scaledLasso(args...; kwargs...) scaled_lasso(args...; kwargs...)
Base.@deprecate scaledLasso!(args...; kwargs...) scaled_lasso!(args...; kwargs...)
Base.@deprecate feasibleLasso!(args...; kwargs...) feasible_lasso!(args...; kwargs...)
Base.@deprecate refitLassoPath(args...; kwargs...) refit_lasso_path(args...; kwargs...)
Base.@deprecate adaptiveLasso(args...; kwargs...) adaptive_lasso(args...; kwargs...)
Base.@deprecate glmLasso(args...; kwargs...) glm_lasso(args...; kwargs...)
Base.@deprecate groupLasso(args...; kwargs...) group_lasso(args...; kwargs...)
Base.@deprecate sparseGroupLasso(args...; kwargs...) sparse_group_lasso(args...; kwargs...)
Base.@deprecate multiLasso(args...; kwargs...) multi_lasso(args...; kwargs...)
Base.@deprecate theoryLambda(args...; kwargs...) theory_lambda(args...; kwargs...)
Base.@deprecate postLasso(args...; kwargs...) post_lasso(args...; kwargs...)
Base.@deprecate doubleSelection(args...; kwargs...) double_selection(args...; kwargs...)
Base.@deprecate varyingCoefficient(args...; kwargs...) varying_coefficient(args...; kwargs...)
Base.@deprecate varyingCoefficientLasso(args...; kwargs...) varying_coefficient_lasso(args...; kwargs...)
Base.@deprecate varyingCoefficientCV(args...; kwargs...) varying_coefficient_cv(args...; kwargs...)
Base.@deprecate varyingCoefficientLassoCV(args...; kwargs...) varying_coefficient_lasso_cv(args...; kwargs...)
Base.@deprecate_binding numCoordinates num_coordinates false
Base.@deprecate_binding descendCoordinate! descend_coordinate! false
Base.@deprecate_binding coordinateScale coordinate_scale false
Base.@deprecate_binding responseScale response_scale false
Base.@deprecate_binding fullGradient! full_gradient! false

# The old solver returned the iterate. Keep that contract for one release;
# the new solver returns CDResults, like ProximalSolvers.solve!.
Base.@deprecate coordinateDescent!(args...; kwargs...) coordinate_descent!(args...; kwargs...).minimizer

function coordinateDescent!(
  x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate, Vector{<:AbstractFloat}},
  f::Union{CDModel, AbstractFunction, CoordinateDifferentiableFunction},
  g::AbstractFunction,
  options::CDOptions,
  info::CDInfo;
  kwargs...)

  Base.depwarn("coordinateDescent!(x, f, g, options, info) is deprecated; use " *
               "r = coordinate_descent!(x, f, g, options) for the iterate and convergence information.",
               :coordinateDescent!)
  _record!(info, solve!(x, f, g, options; kwargs...))
  x
end

const _CD_OPTION_PROPERTIES = (:maxIter => :max_iter, :optTol => :opt_tol,
  :warmStart => :warm_start, :numSteps => :num_steps, :relativeTol => :relative_tol)
const _ITER_OPTION_PROPERTIES = (:maxIter => :max_iter, :optTol => :opt_tol,
  :initProcedure => :init_procedure, :optionsCD => :options_cd)

function _legacy_property(s::Symbol, aliases)
  for (old, new) in aliases
    if s === old
      Base.depwarn("property $old is deprecated; use $new.", old)
      return new
    end
  end
  s
end

@inline function Base.getproperty(o::CDOptions, s::Symbol)
  s === :randomize && return getfield(o, :order) !== :cyclic
  if s === :maxIter
    Base.depwarn("property maxIter is deprecated; use max_iter.", :maxIter)
    return getfield(o, :max_iter)
  end
  if s === :optTol
    Base.depwarn("property optTol is deprecated; use opt_tol.", :optTol)
    return getfield(o, :opt_tol)
  end
  if s === :warmStart
    Base.depwarn("property warmStart is deprecated; use warm_start.", :warmStart)
    return getfield(o, :warm_start)
  end
  if s === :numSteps
    Base.depwarn("property numSteps is deprecated; use num_steps.", :numSteps)
    return getfield(o, :num_steps)
  end
  if s === :relativeTol
    Base.depwarn("property relativeTol is deprecated; use relative_tol.", :relativeTol)
    return getfield(o, :relative_tol)
  end
  getfield(o, s)
end
Base.propertynames(::CDOptions, private::Bool=false) =
  (fieldnames(CDOptions)..., :randomize, first.(_CD_OPTION_PROPERTIES)...)

@inline function Base.getproperty(o::IterLassoOptions, s::Symbol)
  if s === :maxIter
    Base.depwarn("property maxIter is deprecated; use max_iter.", :maxIter)
    return getfield(o, :max_iter)
  end
  if s === :optTol
    Base.depwarn("property optTol is deprecated; use opt_tol.", :optTol)
    return getfield(o, :opt_tol)
  end
  if s === :initProcedure
    Base.depwarn("property initProcedure is deprecated; use init_procedure.", :initProcedure)
    return getfield(o, :init_procedure)
  end
  if s === :optionsCD
    Base.depwarn("property optionsCD is deprecated; use options_cd.", :optionsCD)
    return getfield(o, :options_cd)
  end
  getfield(o, s)
end
Base.propertynames(::IterLassoOptions, private::Bool=false) =
  (fieldnames(IterLassoOptions)..., first.(_ITER_OPTION_PROPERTIES)...)

@inline function Base.getproperty(o::CDInfo, s::Symbol)
  if s === :maxChange
    Base.depwarn("property maxChange is deprecated; use max_change.", :maxChange)
    return getfield(o, :max_change)
  end
  getfield(o, s)
end
function Base.setproperty!(o::CDInfo, s::Symbol, v)
  field = _legacy_property(s, (:maxChange => :max_change,))
  setfield!(o, field, convert(fieldtype(CDInfo, field), v))
end
Base.propertynames(::CDInfo, private::Bool=false) = (fieldnames(CDInfo)..., :maxChange)

for P in (LassoPath, GLMLassoPath, GroupLassoPath, SparseGroupLassoPath, MultiLassoPath)
  @eval begin
    @inline function Base.getproperty(p::$P, s::Symbol)
      if s === :kktChecks
        Base.depwarn("property kktChecks is deprecated; use kkt_checks.", :kktChecks)
        return getfield(p, :kkt_checks)
      end
      getfield(p, s)
    end
    Base.propertynames(p::$P, private::Bool=false) = (fieldnames(typeof(p))..., :kktChecks)
  end
end
