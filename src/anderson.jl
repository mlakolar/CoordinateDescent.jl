# Anderson acceleration of the passes over the active set (Bertrand and Massias,
# Anderson acceleration of coordinate descent, AISTATS 2021).
#
# _coordinateDescent! records the values of the active coordinates after each
# pass over an unchanged active set. After K + 1 such passes x₀, …, x_K it
# extrapolates x_e = Σᵢ cᵢ xᵢ over i = 1, …, K, with the weights c that
# minimize ‖Σᵢ cᵢ (xᵢ - xᵢ₋₁)‖ subject to Σᵢ cᵢ = 1, and keeps x_e if it lowers
# the objective. The history starts again after an extrapolation, after a full
# pass and when the active set shrinks. Extrapolations are often rejected while
# the active set settles, and accepted in runs once it has, so a rejection does
# not stop later attempts.

mutable struct AndersonState{T}
  K::Int
  hist::Matrix{T}       # column j: the active values after the j-th recorded pass
  U::Matrix{T}          # differences of consecutive columns of hist
  xold::Vector{T}       # the active values before an extrapolation, to undo it
  cache::Vector{T}      # the cache of the loss before an extrapolation, to undo it
  m::Int                # number of active coordinates in the history
  filled::Int           # number of passes recorded
end

AndersonState{T}(K::Int) where {T} =
  AndersonState{T}(K, Matrix{T}(undef, 0, K + 1), Matrix{T}(undef, 0, K), T[], T[], 0, 0)

# the state of Anderson acceleration with memory K, or nothing for K = 0
function _andersonState(x, f, g, K::Int)
  K == 0 && return nothing
  x isa SparseIterate ||
    throw(ArgumentError("anderson needs a SparseIterate, not a $(typeof(x))."))
  hasmethod(ProximalBase.value, Tuple{typeof(f), typeof(x)}) ||
    throw(ArgumentError("anderson needs ProximalBase.value(f, x) for the loss, which $(typeof(f)) does not define."))
  g isa Union{ProxL1, ProxZero} || hasmethod(ProximalBase.value, Tuple{typeof(g), typeof(x)}) ||
    throw(ArgumentError("anderson needs ProximalBase.value(g, x) for the penalty, which $(typeof(g)) does not define."))
  AndersonState{eltype(x)}(K)
end

# the penalty at x, from the stored entries of x for ProxL1 and ProxZero
_penaltyValue(g::ProxL1{T, Nothing}, x::SparseIterate) where {T} = g.λ0 * sum(abs, view(x.nzval, 1:nnz(x)))
function _penaltyValue(g::ProxL1{T, S}, x::SparseIterate) where {T, S <: AbstractArray}
  v = zero(T)
  for i in 1:nnz(x)
    v += g.λ[x.nzval2ind[i]] * abs(x.nzval[i])
  end
  g.λ0 * v
end
_penaltyValue(::ProxZero, x::SparseIterate) = zero(eltype(x))
_penaltyValue(g, x) = ProximalBase.value(g, x)

_objective(f, g, x) = ProximalBase.value(f, x) + _penaltyValue(g, x)

# the cache of a built-in loss, copied to undo an extrapolation; the cache of
# another loss is computed again with initialize!
_cache(f::Union{CDLeastSquaresLoss, CDWeightedLSLoss, CDSqrtLassoLoss}) = f.r
_cache(f::CDQuadraticLoss) = f.Ax
_cache(f) = nothing

# records the active values of x after a pass over its active set; returns
# whether K + 1 passes are recorded
function _record!(a::AndersonState{T}, x::SparseIterate{T}) where {T}
  m = nnz(x)
  if m != a.m                   # the active set shrank, or this is the first record
    a.m, a.filled = m, 0
    if size(a.hist, 1) < m
      a.hist = Matrix{T}(undef, m, a.K + 1)
      a.U = Matrix{T}(undef, m, a.K)
      resize!(a.xold, m)
    end
  end
  m == 0 && return false
  a.filled += 1
  copyto!(view(a.hist, 1:m, a.filled), view(x.nzval, 1:m))
  a.filled == a.K + 1
end

# replaces x with the extrapolation from the recorded passes if that lowers the
# objective, and returns whether it did; f must be current for x, and is current
# for x on return
function _extrapolate!(a::AndersonState{T}, x::SparseIterate{T}, f, g) where {T}
  a.filled = 0
  m, K = a.m, a.K
  H = view(a.hist, 1:m, :)
  U = view(a.U, 1:m, :)
  @inbounds for j in 1:K, i in 1:m
    U[i, j] = H[i, j + 1] - H[i, j]
  end
  G = U' * U
  t = tr(G)
  t > 0 && isfinite(t) || return false
  for j in 1:K
    G[j, j] += t * T(1e-12)      # regularizes, as the differences become collinear
  end
  F = cholesky!(Symmetric(G); check = false)
  issuccess(F) || return false
  c = F \ ones(T, K)
  c ./= sum(c)
  all(isfinite, c) || return false

  F0 = _objective(f, g, x)
  xold = view(a.xold, 1:m)
  copyto!(xold, view(x.nzval, 1:m))
  cache = _cache(f)
  cache === nothing || copyto!(resize!(a.cache, length(cache)), cache)
  mul!(view(x.nzval, 1:m), view(H, :, 2:K+1), c)
  initialize!(f, x)
  _objective(f, g, x) < F0 && return true
  copyto!(view(x.nzval, 1:m), xold)
  cache === nothing ? initialize!(f, x) : copyto!(cache, a.cache)
  false
end
