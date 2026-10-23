# Coordinate orders, Anderson acceleration (issue #9) and momentum (issue #20)
# on the lasso
#
#     minimize  ‖y - Xβ‖² / (2n) + λ‖β‖₁,
#
# solved from β = 0 with the default stopping rule (optTol = 1e-7), for
# `order = :cyclic`, `:random` and `:active`, `order = :cyclic` with
# `anderson = 5`, and `order = :random` with `momentum = true`. The designs have
# Gaussian columns scaled to ‖Xⱼ‖² = n, s coefficients ±1 at random positions
# and a signal-to-noise ratio of 4, as in the lasso benchmark of ProximalOPT.jl.
# Prints passes, times and the relative suboptimality of the solution, then the
# times of 50-λ paths (`LassoPath`, λmax to λmax/100) and a comparison of
# memories K. Other processes load the machine unevenly, so every case runs the
# solvers in turn, until each has at least half a second of runs and at least 3
# runs (5 for the random orders); the times are the minimum for the
# deterministic orders, and the passes and times the median for the random ones.
#
#     julia --project=benchmark benchmark/order_anderson_bench.jl

using CoordinateDescent, ProximalBase, LinearAlgebra, Random, Statistics, Printf
BLAS.set_num_threads(1)

function design(; n, p, s, ρ = 0.0, common = 0.0, factors = 0, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if ρ > 0
    c = sqrt(1 - ρ^2)
    for j in 2:p
      @views X[:, j] .= ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  common > 0 && (X .= sqrt(1 - common) .* X .+ sqrt(common) .* randn(rng, n))
  factors > 0 && (X .+= randn(rng, n, factors) * randn(rng, factors, p))
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  X, signal .+ sqrt(sum(abs2, signal) / n / 4) .* randn(rng, n)
end

const DESIGNS = [
  ("A: n=500 p=2500, independent",         (n = 500,  p = 2500,  s = 50)),
  ("B: n=500 p=2500, AR(1) 0.8",           (n = 500,  p = 2500,  s = 50, ρ = 0.8)),
  ("B': n=500 p=2500, AR(1) 0.95",         (n = 500,  p = 2500,  s = 50, ρ = 0.95)),
  ("C: n=5000 p=500, AR(1) 0.5",           (n = 5000, p = 500,   s = 50, ρ = 0.5)),
  ("D: n=200 p=10000, independent",        (n = 200,  p = 10000, s = 20)),
  ("E: n=500 p=2500, equicorrelated 0.5",  (n = 500,  p = 2500,  s = 50, common = 0.5)),
  ("E': n=500 p=2500, equicorrelated 0.9", (n = 500,  p = 2500,  s = 50, common = 0.9)),
  ("F: n=500 p=2500, 3 factors",           (n = 500,  p = 2500,  s = 50, factors = 3)),
]

const SOLVERS = [          # name, options, random order
  ("cyclic",   CDOptions(; order = :cyclic),                 false),
  ("random",   CDOptions(; order = :random),                 true),
  ("active",   CDOptions(; order = :active),                 true),
  ("Anderson", CDOptions(; order = :cyclic, anderson = 5),    false),
  ("momentum", CDOptions(; order = :random, momentum = true), true),
]

objective(X, y, β, λ) = sum(abs2, y - X * β) / (2 * size(X, 1)) + λ * sum(abs, β)

function solve(X, y, λ, opts)
  x = SparseIterate(size(X, 2))
  info = CDInfo()
  coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), opts, info)
  x, info.iterations
end

med(v) = sort(v)[(length(v) + 1) ÷ 2]

# runs run(opts) for the options of every solver in turn, after one run each to
# compile, until each has at least half a second of runs and at least 3 runs (5
# with a random order); returns for each solver its results and times
function interleaved(run, solvers)
  for (_, opts, _) in solvers
    run(opts)
  end
  results = [Any[] for _ in solvers]
  times = [Float64[] for _ in solvers]
  enough(j) = length(times[j]) >= (solvers[j][3] ? 5 : 3) && sum(times[j]) >= 0.5
  while !all(enough, eachindex(solvers))
    for (j, (_, opts, _)) in enumerate(solvers)
      enough(j) && continue
      t = @elapsed r = run(opts)
      push!(results[j], r)
      push!(times[j], t)
    end
  end
  results, times
end

# the time of a solver: the median with a random order, otherwise the minimum
summarize(ts, random) = random ? med(ts) : minimum(ts)

println("# One λ, from β = 0, optTol = 1e-7: milliseconds (passes) and (F - F*)/F*\n")
println("| design | λ/λmax | ", join(first.(SOLVERS), " | "), " |")
println("|---|---:|", repeat("---:|", length(SOLVERS)))
for (name, kw) in DESIGNS
  X, y = design(; kw...)
  n = size(X, 1)
  λmax = maximum(abs, X' * y) / n
  for r in (0.1, 0.02, 0.01)
    λ = r * λmax
    xref, _ = solve(X, y, λ, CDOptions(; order = :random, optTol = 1e-13, maxIter = 10^7))
    Fstar = objective(X, y, Vector(xref), λ)
    results, times = interleaved(opts -> solve(X, y, λ, opts), SOLVERS)
    cells = String[]
    for (j, (_, _, random)) in enumerate(SOLVERS)
      passes = [res[2] for res in results[j]]
      gap = maximum(max(objective(X, y, Vector(res[1]), λ) - Fstar, 0) / Fstar for res in results[j])
      push!(cells, @sprintf("%.1f (%d, %.0e)", 1e3 * summarize(times[j], random),
                            random ? med(passes) : passes[argmin(times[j])], gap))
    end
    println("| ", name, " | ", r, " | ", join(cells, " | "), " |")
  end
end

println("\n# 50-λ path from λmax to λmax/100 (LassoPath, standardizeX = false): milliseconds\n")
println("| design | ", join(first.(SOLVERS), " | "), " |")
println("|---|", repeat("---:|", length(SOLVERS)))
for (name, kw) in DESIGNS[[1, 2, 5, 6, 8]]
  X, y = design(; kw...)
  _, times = interleaved(opts -> LassoPath(X, y, opts; nλ = 50, λminratio = 0.01, standardizeX = false), SOLVERS)
  cells = [@sprintf("%.0f", 1e3 * summarize(times[j], SOLVERS[j][3])) for j in eachindex(SOLVERS)]
  println("| ", name, " | ", join(cells, " | "), " |")
end

println("\n# Memory K of Anderson acceleration at λ = 0.02 λmax: milliseconds (passes)\n")
println("| design | K = 3 | K = 5 | K = 10 |")
println("|---|---:|---:|---:|")
KS = [("K = $K", CDOptions(; order = :cyclic, anderson = K), false) for K in (3, 5, 10)]
for (name, kw) in DESIGNS[[1, 2, 5]]
  X, y = design(; kw...)
  λ = 0.02 * maximum(abs, X' * y) / size(X, 1)
  results, times = interleaved(opts -> solve(X, y, λ, opts), KS)
  cells = [@sprintf("%.1f (%d)", 1e3 * minimum(times[j]), results[j][argmin(times[j])][2]) for j in eachindex(KS)]
  println("| ", name, " | ", join(cells, " | "), " |")
end
