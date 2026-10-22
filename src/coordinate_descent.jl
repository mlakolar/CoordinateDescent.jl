"""
    coordinateDescent!(x, f, g, options=CDOptions(), info=CDInfo())

Minimize `f(x) + g(x)` by coordinate descent and store the solution in `x`,
where `f` is a [`CoordinateDifferentiableFunction`](@ref), `g` is a
[`ProxL1`](@ref ProximalBase.ProxL1) or [`ProxZero`](@ref ProximalBase.ProxZero)
penalty and `x` is a `SparseIterate`.

The method alternates between full passes over all coordinates and passes over
the nonzero coordinates only (the active set). It stops when a full pass is
followed by a pass in which no coordinate changes by more than
`options.optTol`, or after `options.maxIter` passes. In the second case `x` may
be far from the minimizer, and a warning is issued.

If `options.warmStart` is true, the descent starts from the supplied `x`.
Otherwise `x` is set to zero and, for `ProxL1`, the problem is solved along a
geometric sequence of `options.numSteps` penalty levels that decreases from the
smallest `λ0` with zero solution to `g.λ0`, each one warm started from the
previous solution. Coordinates with zero penalty weight are left out when
finding that `λ0`.

The number of passes and whether the stopping rule was met are stored in `info`
(see [`CDInfo`](@ref)). Returns `x`.

A dense `Vector` `x` is also accepted; it is converted to a `SparseIterate` and
overwritten with the solution.
"""
function coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::ProxL1,
  options::CDOptions=CDOptions(),
  info::CDInfo=CDInfo())

  ProximalBase.numCoordinates(x) == numCoordinates(f) || throw(DimensionMismatch())
  if !isa(g, ProxL1{typeof(g.λ0), Nothing})
    length(g.λ) == numCoordinates(f) || throw(DimensionMismatch())
  end

  coef_iterator = options.randomize ? RandomIterator(x) : OrderedIterator(x)

  if options.warmStart
    initialize!(f, x)
    _coordinateDescent!(x, f, g, coef_iterator, options, info)
  else
    # set x to zero and initialize
    fill!(x, zero(eltype(x)))
    initialize!(f, x)

    # find λmax
    λmax = _findLambdaMax(x, f, g)

    # find decreasing schedule for λ, unless g.λ0 is zero or at least λmax;
    # the last step is solved at g itself, since a floating point range
    # l1:step:l2 may stop short of l2
    passes = 0
    if zero(λmax) < g.λ0 < λmax
      l1, l2 = log(λmax), log(g.λ0)
      for k in 0:options.numSteps-1
        g1 = ProxL1(exp(l1 + k * (l2 - l1) / options.numSteps), g.λ)
        _coordinateDescent!(x, f, g1, coef_iterator, options, info)
        passes += info.iterations
      end
    end
    _coordinateDescent!(x, f, g, coef_iterator, options, info)
    info.iterations += passes
  end
  _warn_not_converged(info, options)
  x
end

function coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::ProxZero,
  options::CDOptions=CDOptions(),
  info::CDInfo=CDInfo())

  ProximalBase.numCoordinates(x) == numCoordinates(f) || throw(DimensionMismatch())

  coef_iterator = options.randomize ? RandomIterator(x) : OrderedIterator(x)

  if options.warmStart
    initialize!(f, x)
  else
    # set x to zero and initialize
    fill!(x, zero(eltype(x)))
    initialize!(f, x)
  end

  _coordinateDescent!(x, f, g, coef_iterator, options, info)
  _warn_not_converged(info, options)
  x
end

function coordinateDescent!(
  x::Vector{T},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  options::CDOptions=CDOptions(),
  info::CDInfo=CDInfo()) where {T<:AbstractFloat}

  xs = SparseIterate(x)
  coordinateDescent!(xs, f, g, options, info)
  copyto!(x, xs)
end



# Alternates full passes with passes over the active set; stops after two
# consecutive passes below optTol, the first of which is a full pass, so that
# the active set is checked against all coordinates before stopping.
# Records the passes made and whether it stopped in info.
# Assumes that f is initialized before the call here.
function _coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  coef_iterator::AtomIterator,
  options::CDOptions,
  info::CDInfo)

  prev_converged = false
  converged = true
  info.iterations, info.converged, info.maxChange = 0, false, NaN
  for iter=1:options.maxIter

    # a full pass after convergence on the active set, otherwise a pass over
    # the nonzero coordinates
    reset!(coef_iterator, converged)
    maxH = _cdPass!(x, f, g, coef_iterator)
    prev_converged = converged
    info.iterations, info.maxChange = iter, maxH

    # test for convergence
    converged = maxH < options.optTol

    if prev_converged && converged
      info.converged = true
      break
    end
  end
  x
end

_warn_not_converged(info::CDInfo, options::CDOptions) =
  info.converged || @warn "coordinate descent did not converge in maxIter = $(options.maxIter) passes (largest change of a coordinate in the last pass $(round(info.maxChange; sigdigits = 3)) > optTol = $(options.optTol)); the solution may be inaccurate. Increase maxIter in CDOptions."

function _cdPass!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  coef_iterator::AtomIterator
  )

  maxH = zero(eltype(x))
  for ipred = coef_iterator               # coef_iterator produces original indexes
    h = descendCoordinate!(f, g, x, ipred)
    if abs(h) > maxH
      maxH = abs(h)
    end
  end
  dropzeros!(x)
  maxH
end


######

# the smallest λ0 for which zero is a solution: max_k |∇f(0)_k|
# (x must be zero when this is called)
function _findLambdaMax(x::Union{SparseIterate{T},SymmetricSparseIterate{T}},
  f::CoordinateDifferentiableFunction,
  ::ProxL1{T, Nothing}) where {T<:AbstractFloat}

  λmax = zero(T)
  for k=1:ProximalBase.numCoordinates(x)
    f_g = gradient(f, x, k)
    t = abs(f_g)
    if t > λmax
      λmax = t
    end
  end
  λmax
end

# the smallest λ0 for which zero is a solution with weights λ: max_k |∇f(0)_k| / λ_k
# over the penalized coordinates; those with λ_k = 0 are not zero at any λ0
function _findLambdaMax(x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  g::ProxL1{T, S}) where {T<:AbstractFloat} where S <: AbstractArray

  λmax = zero(T)
  for k=1:length(x)
    iszero(g.λ[k]) && continue
    f_g = gradient(f, x, k)
    t = abs(f_g) / g.λ[k]
    if t > λmax
      λmax = t
    end
  end
  λmax
end
