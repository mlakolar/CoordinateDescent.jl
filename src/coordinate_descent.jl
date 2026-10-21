"""
    coordinateDescent!(x, f, g, options=CDOptions())

Minimize `f(x) + g(x)` by coordinate descent and store the solution in `x`,
where `f` is a [`CoordinateDifferentiableFunction`](@ref), `g` is a
[`ProxL1`](@ref ProximalBase.ProxL1) or [`ProxZero`](@ref ProximalBase.ProxZero)
penalty and `x` is a `SparseIterate`.

The method alternates between full passes over all coordinates and passes over
the nonzero coordinates only (the active set). It stops when a full pass is
followed by a pass in which no coordinate changes by more than
`options.optTol`, or after `options.maxIter` passes.

If `options.warmStart` is true, the descent starts from the supplied `x`.
Otherwise `x` is set to zero and, for `ProxL1`, the problem is solved along a
geometric sequence of `options.numSteps` penalty levels that decreases from the
smallest `λ0` with zero solution to `g.λ0`, each one warm started from the
previous solution.

Returns `x`.
"""
function coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::ProxL1,
  options::CDOptions=CDOptions())

  ProximalBase.numCoordinates(x) == numCoordinates(f) || throw(DimensionMismatch())
  if !isa(g, ProxL1{typeof(g.λ0), Nothing})
    length(g.λ) == numCoordinates(f) || throw(DimensionMismatch())
  end

  coef_iterator = options.randomize ? RandomIterator(x) : OrderedIterator(x)

  if options.warmStart
    initialize!(f, x)
    return _coordinateDescent!(x, f, g, coef_iterator, options)
  else
    # set x to zero and initialize
    fill!(x, zero(eltype(x)))
    initialize!(f, x)

    # find λmax
    λmax = _findLambdaMax(x, f, g)

    # find decreasing schedule for λ; the last step is solved at g itself,
    # since a floating point range l1:step:l2 may stop short of l2
    l1, l2 = log(λmax), log(g.λ0)
    for k in 0:options.numSteps-1
      g1 = ProxL1(exp(l1 + k * (l2 - l1) / options.numSteps), g.λ)
      _coordinateDescent!(x, f, g1, coef_iterator, options)
    end
    return _coordinateDescent!(x, f, g, coef_iterator, options)
  end
end

function coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::ProxZero,
  options::CDOptions=CDOptions())

  ProximalBase.numCoordinates(x) == numCoordinates(f) || throw(DimensionMismatch())

  coef_iterator = options.randomize ? RandomIterator(x) : OrderedIterator(x)

  if options.warmStart
    initialize!(f, x)
  else
    # set x to zero and initialize
    fill!(x, zero(eltype(x)))
    initialize!(f, x)
  end

  return _coordinateDescent!(x, f, g, coef_iterator, options)
end



# Alternates full passes with passes over the active set; stops after two
# consecutive passes below optTol, the first of which is a full pass, so that
# the active set is checked against all coordinates before stopping.
# Assumes that f is initialized before the call here.
function _coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  coef_iterator::AtomIterator,
  options::CDOptions)

  prev_converged = false
  converged = true
  for iter=1:options.maxIter

    # a full pass after convergence on the active set, otherwise a pass over
    # the nonzero coordinates
    reset!(coef_iterator, converged)
    maxH = _cdPass!(x, f, g, coef_iterator)
    prev_converged = converged

    # test for convergence
    converged = maxH < options.optTol

    prev_converged && converged && break
  end
  x
end

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
function _findLambdaMax(x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  g::ProxL1{T, S}) where {T<:AbstractFloat} where S <: AbstractArray

  λmax = zero(T)
  for k=1:length(x)
    f_g = gradient(f, x, k)
    t = abs(f_g) / g.λ[k]
    if t > λmax
      λmax = t
    end
  end
  λmax
end
