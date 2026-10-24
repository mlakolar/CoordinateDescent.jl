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
function _coordinate_descent_iterate!(x, args...; kwargs...)
  coordinate_descent!(x, args...; kwargs...)
  x
end
Base.@deprecate coordinateDescent!(args...; kwargs...) _coordinate_descent_iterate!(args...; kwargs...)

function _coordinate_descent_iterate!(
  x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate, Vector{<:AbstractFloat}},
  f::Union{CDModel, AbstractFunction, CoordinateDifferentiableFunction},
  g::AbstractFunction,
  options::CDOptions,
  info::CDInfo;
  kwargs...)

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

Base.getproperty(o::CDOptions, s::Symbol) = s === :randomize ? getfield(o, :order) !== :cyclic :
  getfield(o, _legacy_property(s, _CD_OPTION_PROPERTIES))
Base.propertynames(::CDOptions, private::Bool=false) =
  (fieldnames(CDOptions)..., :randomize, first.(_CD_OPTION_PROPERTIES)...)
Base.getproperty(o::IterLassoOptions, s::Symbol) = getfield(o, _legacy_property(s, _ITER_OPTION_PROPERTIES))
Base.propertynames(::IterLassoOptions, private::Bool=false) =
  (fieldnames(IterLassoOptions)..., first.(_ITER_OPTION_PROPERTIES)...)
Base.getproperty(o::CDInfo, s::Symbol) = getfield(o, _legacy_property(s, (:maxChange => :max_change,)))
function Base.setproperty!(o::CDInfo, s::Symbol, v)
  field = _legacy_property(s, (:maxChange => :max_change,))
  setfield!(o, field, convert(fieldtype(CDInfo, field), v))
end
Base.propertynames(::CDInfo, private::Bool=false) = (fieldnames(CDInfo)..., :maxChange)

for P in (LassoPath, GLMLassoPath, GroupLassoPath, SparseGroupLassoPath, MultiLassoPath)
  @eval begin
    Base.getproperty(p::$P, s::Symbol) = getfield(p, _legacy_property(s, (:kktChecks => :kkt_checks,)))
    Base.propertynames(p::$P, private::Bool=false) = (fieldnames(typeof(p))..., :kktChecks)
  end
end
