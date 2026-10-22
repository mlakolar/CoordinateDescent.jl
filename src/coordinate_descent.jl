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
finding that `λ0`. For a `SparseIterate`, each penalty level is screened with
the sequential strong rule, as in [`LassoPath`](@ref).

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
    # l1:step:l2 may stop short of l2. For a SparseIterate, every step is
    # screened by the strong rule
    passes = 0
    if zero(λmax) < g.λ0 < λmax
      l1, l2 = log(λmax), log(g.λ0)
      screen = x isa SparseIterate
      if screen
        grad = Vector{eltype(x)}(undef, ProximalBase.numCoordinates(x))
        fullGradient!(grad, f, x)
        inS = falses(length(grad))
        λprev = λmax
      end
      for k in 0:options.numSteps-1
        g1 = ProxL1(exp(l1 + k * (l2 - l1) / options.numSteps), g.λ)
        if screen
          _screenedSolve!(x, f, g1, λprev, grad, coef_iterator, options, info, inS)
          λprev = g1.λ0
        else
          _coordinateDescent!(x, f, g1, coef_iterator, options, info)
        end
        passes += info.iterations
      end
      if screen
        _screenedSolve!(x, f, g, λprev, grad, coef_iterator, options, info, inS)
      else
        _coordinateDescent!(x, f, g, coef_iterator, options, info)
      end
    else
      _coordinateDescent!(x, f, g, coef_iterator, options, info)
    end
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
# consecutive passes below the tolerance, the first of which is a full pass, so
# that the active set is checked against all coordinates before stopping.
# With options.relativeTol the changes are scaled by coordinateScale and the
# tolerance by responseScale; a zero response scale falls back to the absolute
# rule. Records the passes made and whether it stopped in info.
# Assumes that f is initialized before the call here.
function _coordinateDescent!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  coef_iterator::AtomIterator,
  options::CDOptions,
  info::CDInfo)

  sy = options.relativeTol ? responseScale(f) : 1
  scaled = options.relativeTol && sy > 0
  tol = scaled ? options.optTol * sy : options.optTol

  prev_converged = false
  converged = true
  info.iterations, info.converged, info.maxChange, info.tolerance = 0, false, NaN, tol
  for iter=1:options.maxIter

    # a full pass after convergence on the active set, otherwise a pass over
    # the nonzero coordinates
    reset!(coef_iterator, converged)
    maxH = _cdPass!(x, f, g, coef_iterator, scaled)
    prev_converged = converged
    info.iterations, info.maxChange = iter, maxH

    # test for convergence
    converged = maxH < tol

    if prev_converged && converged
      info.converged = true
      break
    end
  end
  x
end

# Solves at the penalty g, warm started from x, with the sequential strong rule
# of Tibshirani et al. (2012) for the step from the penalty level λprev to
# λ = g.λ0: full passes visit only the coordinates with
# |∇ₖf| ≥ (2λ - λprev) ωₖ at the previous solution, the nonzero ones and the
# unpenalized ones. After convergence the optimality conditions |∇ₖf| ≤ λ ωₖ of
# the other coordinates are checked with one full gradient, and violators are
# added and the problem solved again. f must be current for x, and grad holds
# the gradient at x on entry and on return. Returns the number of checks.
function _screenedSolve!(
  x::SparseIterate{T},
  f::CoordinateDifferentiableFunction,
  g::ProxL1{T},
  λprev::T,
  grad::Vector{T},
  coef_iterator::AtomIterator,
  options::CDOptions,
  info::CDInfo,
  inS::BitVector) where {T<:AbstractFloat}

  λ = g.λ0
  threshold = 2λ - λprev
  for k in eachindex(grad)
    ω = _weight(g, k)
    inS[k] = iszero(ω) || !iszero(x[k]) || abs(grad[k]) >= threshold * ω
  end
  passes, checks = 0, 0
  while true
    setCandidates!(coef_iterator, findall(inS))
    _coordinateDescent!(x, f, g, coef_iterator, options, info)
    passes += info.iterations
    fullGradient!(grad, f, x)
    checks += 1
    violated = false
    for k in eachindex(grad)
      if !inS[k] && abs(grad[k]) > λ * _weight(g, k)
        inS[k] = true
        violated = true
      end
    end
    violated || break
  end
  setCandidates!(coef_iterator, nothing)
  info.iterations = passes
  checks
end

_weight(::ProxL1{T, Nothing}, ::Int) where {T} = one(T)
_weight(g::ProxL1{T, S}, k::Int) where {T, S <: AbstractArray} = g.λ[k]

_warn_not_converged(info::CDInfo, options::CDOptions) =
  info.converged || @warn "coordinate descent did not converge in maxIter = $(options.maxIter) passes (largest change of a coordinate in the last pass $(round(info.maxChange; sigdigits = 3)) > tolerance $(round(info.tolerance; sigdigits = 3))); the solution may be inaccurate. Increase maxIter in CDOptions."

function _cdPass!(
  x::Union{SparseIterate,SymmetricSparseIterate,AtomIterate},
  f::CoordinateDifferentiableFunction,
  g::Union{ProxL1, ProxZero},
  coef_iterator::AtomIterator,
  scaled::Bool
  )

  maxH = zero(eltype(x))
  for ipred = coef_iterator               # coef_iterator produces original indexes
    h = descendCoordinate!(f, g, x, ipred)
    ah = scaled ? oftype(maxH, abs(h) * coordinateScale(f, ipred)) : oftype(maxH, abs(h))
    if ah > maxH
      maxH = ah
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
