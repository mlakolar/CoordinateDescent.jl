# refit_lasso_path with the factor updated along the path (ProximalSolvers'
# RefitCache) against the previous implementation, a pivoted QR of X[:, S] for
# every support, on the same paths. One BLAS thread; run it on Discovery.
#
#     julia --project=. benchmark/refit_bench.jl

using HighDimensionalModels, LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)

# the implementation before RefitCache
function refit_qr(path, X, Y)
  out = Dict{Vector{Int},Vector{Float64}}()
  for β in path.βpath
    S = findall(!iszero, β)
    haskey(out, S) && continue
    out[S] = path.intercept ? [ones(length(Y)) X[:, S]] \ Y : X[:, S] \ Y
  end
  out
end

function design(n, p, s; ρ = 0.0, seed = 1)
  rng = MersenneTwister(seed)
  X = randn(rng, n, p)
  if ρ > 0
    for j in 2:p
      X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
    end
  end
  β = zeros(p); β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  X, X * β + randn(rng, n)
end

mintime(f; reps = 5) = minimum(@elapsed(f()) for _ in 1:reps)

println("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", BLAS threads = ", BLAS.get_num_threads())
@printf "%-34s %8s %6s %10s %10s %7s %10s\n" "case" "supports" "max|S|" "QR (ms)" "cache (ms)" "ratio" "max diff"
for (name, n, p, s, ρ, intercept) in (("n=500 p=2500 independent", 500, 2500, 50, 0.0, false),
                                      ("n=500 p=2500 AR(1) 0.8", 500, 2500, 50, 0.8, false),
                                      ("n=5000 p=500 AR(1) 0.5", 5000, 500, 50, 0.5, false),
                                      ("n=1000 p=5000 independent, intercept", 1000, 5000, 100, 0.0, true))
  X, Y = design(n, p, s; ρ)
  path = LassoPath(X, Y; nλ = 100, λminratio = 0.01, intercept)
  r1, r2 = refit_qr(path, X, Y), refit_lasso_path(path, X, Y)     # compile, and check
  diff = maximum(maximum(abs, r1[S] - r2[S]) / max(maximum(abs, r1[S]), 1.0) for S in keys(r1) if !isempty(S); init = 0.0)
  # interleave the two, keeping each one's fastest run
  t1, t2 = Inf, Inf
  for _ in 1:3
    t1 = min(t1, mintime(() -> refit_qr(path, X, Y)))
    t2 = min(t2, mintime(() -> refit_lasso_path(path, X, Y)))
  end
  @printf "%-34s %8d %6d %10.1f %10.1f %7.2f %10.1e\n" name length(r1) maximum(length, keys(r1)) 1e3t1 1e3t2 t1 / t2 diff
end
