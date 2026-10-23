# Nesterov momentum on the passes over the active set, for any order of the
# coordinates; Xu and Yin (2013) extrapolate block coordinate descent in the
# same way.
#
# Before a pass over the active set, _coordinateDescent! moves the active
# coordinates to y = xₖ + βₖ (xₖ - xₖ₋₁), with βₖ = (k - 1) / (k + 2), and moves
# the cache of the loss in the same way, which is exact because the caches of
# the built-in losses are affine in x; the pass then starts from y. Other losses
# are initialized at y instead. The momentum restarts (k = 1) when the step of
# a pass points against it, (xₖ₊₁ - y)'(xₖ - xₖ₋₁) < 0 (O'Donoghue and Candès,
# 2015), after a full pass, and when the active set shrinks.

mutable struct MomentumState{T}
  xprev::Vector{T}      # the active values after the pass before the last
  xcur::Vector{T}       # the active values after the last pass
  y::Vector{T}          # the active values the last pass started from
  cprev::Vector{T}      # the cache of the loss at xprev
  ccur::Vector{T}       # the cache of the loss at xcur
  m::Int                # number of active coordinates in the history
  k::Int
  moved::Bool           # whether the cache moved since it was last computed
end

MomentumState{T}() where {T} = MomentumState{T}(T[], T[], T[], T[], T[], 0, 1, false)

# the state of the momentum, or nothing without it. By default (momentum =
# nothing) it is used where it is known to help: with a random order, a
# SparseIterate and a loss whose cache moves with the point, the built-in ones;
# initializing another loss at every extrapolated point can cost more than the
# momentum saves
function _momentumState(x, f, options::CDOptions)
  momentum = options.momentum
  momentum === false && return nothing
  if momentum === nothing
    options.order !== :cyclic && x isa SparseIterate && _cache(f) !== nothing || return nothing
  else
    x isa SparseIterate ||
      throw(ArgumentError("momentum needs a SparseIterate, not a $(typeof(x))."))
  end
  MomentumState{eltype(x)}()
end

# starts the history at x, as after a full pass or when the active set shrank
function _momentumReset!(s::MomentumState, x::SparseIterate, f)
  m = nnz(x)
  s.m, s.k = m, 1
  copyto!(resize!(s.xcur, m), view(x.nzval, 1:m))
  copyto!(resize!(s.xprev, m), s.xcur)
  resize!(s.y, m)
  c = _cache(f)
  if c !== nothing
    copyto!(resize!(s.ccur, length(c)), c)
    copyto!(resize!(s.cprev, length(c)), c)
  end
  s
end

# moves x and the cache of f to the point the next pass over the active set
# starts from
function _momentumStep!(s::MomentumState{T}, x::SparseIterate{T}, f) where {T}
  m = s.m
  β = s.k > 1 ? T((s.k - 1) / (s.k + 2)) : zero(T)
  @inbounds for j in 1:m
    s.y[j] = s.xcur[j] + β * (s.xcur[j] - s.xprev[j])
  end
  iszero(β) && return s
  copyto!(view(x.nzval, 1:m), s.y)
  c = _cache(f)
  if c === nothing
    initialize!(f, x)
  else
    @inbounds @simd for i in eachindex(c)
      c[i] = s.ccur[i] + β * (s.ccur[i] - s.cprev[i])
    end
    s.moved = true
  end
  s
end

# after a pass over the active set: restarts when its step points against the
# momentum, and records the new point
function _momentumUpdate!(s::MomentumState{T}, x::SparseIterate{T}, f) where {T}
  nnz(x) == s.m || return _momentumReset!(s, x, f)
  m = s.m
  d = zero(T)
  @inbounds for j in 1:m
    d += (x.nzval[j] - s.y[j]) * (s.xcur[j] - s.xprev[j])
  end
  s.k = s.k > 1 && d < 0 ? 1 : s.k + 1
  s.xprev, s.xcur = s.xcur, s.xprev
  copyto!(s.xcur, view(x.nzval, 1:m))
  c = _cache(f)
  if c !== nothing
    s.cprev, s.ccur = s.ccur, s.cprev
    copyto!(s.ccur, c)
  end
  s
end

# computes the cache of f again where repeated moves may have let it drift
function _momentumRefresh!(s::MomentumState, x, f)
  s.moved && initialize!(f, x)
  s.moved = false
  s
end
