"""
    stability_bound(p, q, cutoff; theta=q/p)

Distribution-free complementary-pairs bound `theta*q/(2cutoff-1)` of Shah
and Samworth (2013, Theorem 1) on the expected number of selected variables
whose half-sample selection probabilities are at most theta. q is an upper
bound on the expected base selection size. With `theta=q/p`, this is
`q²/(p*(2cutoff-1))`, also used by stabs with `assumption="none"`.
It bounds expected false selections (PFER) only if every noise variable has
selection probability at most theta. It is not a general FDR guarantee;
exchangeability/no-worse-than-random assumptions can justify theta=q/p.
"""
function stability_bound(p::Integer, q::Real, cutoff::Real; theta::Real=q/p)
  p > 0 && isfinite(q) && 0 <= q <= p || throw(ArgumentError("require p > 0 and 0 ≤ q ≤ p"))
  isfinite(cutoff) && 0.5 < cutoff <= 1 || throw(ArgumentError("cutoff must be in (0.5, 1]"))
  isfinite(theta) && 0 <= theta <= 1 || throw(ArgumentError("theta must be in [0, 1]"))
  theta * q / (2cutoff - 1)
end

"""
    complementary_pairs(n, npairs; rng=Random.default_rng())

Independent random pairs of disjoint subsets of `1:n`, each of size floor(n/2).
For odd n, one randomly chosen observation per pair is unused. Drawn before
parallel fitting, so results do not depend on thread scheduling.
"""
function complementary_pairs(n::Integer, npairs::Integer; rng::AbstractRNG=default_rng())
  n >= 2 && npairs > 0 || throw(ArgumentError("n must be at least 2 and npairs positive"))
  h = n ÷ 2
  [let perm = randperm(rng, n); (perm[1:h], perm[h+1:2h]) end for _ in 1:npairs]
end

function _stability_pairs(pairs, n)
  isempty(pairs) && throw(ArgumentError("at least one complementary pair is required"))
  out = Tuple{Vector{Int}, Vector{Int}}[]
  for pair in pairs
    length(pair) == 2 || throw(ArgumentError("each pair must have two subsamples"))
    a, b = collect(Int, pair[1]), collect(Int, pair[2])
    length(a) == length(b) == n ÷ 2 || throw(ArgumentError("subsamples must have floor(n/2) rows"))
    ind = [a; b]
    all(i -> 1 <= i <= n, ind) && length(unique(ind)) == length(ind) ||
      throw(ArgumentError("complementary subsamples must have distinct valid row indices"))
    push!(out, (a, b))
  end
  out
end

"""
    StabilitySelectionResult

Result of [`stability_selection`](@ref). `frequencies` averages 2*npairs base
selection indicators; `pair_frequencies` averages simultaneous selection in
both halves. `selected` lists frequencies ≥ cutoff. `selections` stores one
Boolean column per half; `pairs` records the rows. `q` is the deterministic
base selection cap and `pfer_bound` is [`stability_bound`](@ref), under its
stated assumptions. `average_size` is descriptive, not substituted into the bound.
"""
struct StabilitySelectionResult
  selected::Vector{Int}
  frequencies::Vector{Float64}
  pair_frequencies::Vector{Float64}
  selections::Matrix{Bool}
  pairs::Vector{Tuple{Vector{Int}, Vector{Int}}}
  q::Int
  cutoff::Float64
  pfer_bound::Float64
  average_size::Float64
end

"""
    stability_selection(X, y, lambda; q, cutoff=0.9, npairs=50, pairs=nothing,
                        parallel=true, intercept=true, standardize_x=true,
                        rng=Random.default_rng(), options=CDOptions(), kwargs...)

Complementary-pairs stability selection with a lasso base selector at a fixed
positive lambda. On each half, retain at most q nonzero lasso coefficients,
ordered by magnitude on the fitted (standardized, when requested) scale.
This explicit cap makes q a valid upper bound on the base selection size.
An index breaks exact magnitude ties. This base selector differs from stopping
an entire lasso path after q variables enter. Supply `pairs` for shared-data
comparisons; otherwise draw npairs with [`complementary_pairs`](@ref).

Fits run on Julia threads when `parallel=true`; use `julia --threads=N`.
Each fit has its own pre-drawn RNG seed and output column, giving the same
result serially and in parallel. Nonconvergence raises an error instead of
silently removing a failed half. `kwargs` are CDOptions fields. For the
interpretation and assumptions of `pfer_bound`, see [`stability_bound`](@ref).
"""
function stability_selection(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::Real;
                              q::Integer, cutoff::Real=0.9, npairs::Integer=50, pairs=nothing,
                              parallel::Bool=true, intercept::Bool=true, standardize_x::Bool=true,
                              rng::AbstractRNG=default_rng(), options::CDOptions=CDOptions(), kwargs...)
  n, p = size(X)
  n >= 2 && p > 0 || throw(ArgumentError("X needs at least two rows and one column"))
  length(y) == n || throw(DimensionMismatch("X and y have different observation counts"))
  all(isfinite, X) && all(isfinite, y) || throw(ArgumentError("X and y must be finite"))
  isfinite(lambda) && lambda > 0 || throw(ArgumentError("lambda must be finite and positive"))
  bound = stability_bound(p, q, cutoff)
  ps = _stability_pairs(pairs === nothing ? complementary_pairs(n, npairs; rng) : pairs, n)
  B = length(ps)
  opts = _options(options, kwargs)
  seeds = rand(rng, UInt64, 2B)
  selected = zeros(Bool, p, 2B) # Matrix, not BitMatrix: concurrent columns must not share bit words.
  function fit_half(k)
    rows = ps[(k+1)÷2][isodd(k) ? 1 : 2]
    xx, yy = X[rows, :], y[rows]
    sol = lasso(xx, yy, lambda, opts; intercept, standardize_x, rng=Xoshiro(seeds[k]))
    sol.converged || error("stability lasso failed to converge on half $k")
    b = abs.(Vector(sol.x))
    standardize_x && (b .*= _loadings(float.(xx), intercept))
    order = sortperm(b; rev=true) # stable index tie breaking
    for j in order[1:min(q, count(>(0), b))]
      selected[j, k] = true
    end
    nothing
  end
  if parallel && Threads.nthreads() > 1
    Threads.@threads for k in 1:2B
      fit_half(k)
    end
  else
    foreach(fit_half, 1:2B)
  end
  freq = vec(sum(selected; dims=2)) ./ (2B)
  joint = vec(sum(selected[:, 1:2:2B] .& selected[:, 2:2:2B]; dims=2)) ./ B
  StabilitySelectionResult(findall(>=(cutoff), freq), freq, joint, selected, ps,
    q, cutoff, bound, sum(freq))
end
