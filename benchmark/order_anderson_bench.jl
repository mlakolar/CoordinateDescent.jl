# Coordinate orders and Anderson acceleration (issue #9) on the lasso
#
#     minimize  ‖y - Xβ‖² / (2n) + λ‖β‖₁,
#
# solved from β = 0 with the default stopping rule (optTol = 1e-7), for
# `order = :cyclic`, `:random` and `:active`, and `order = :cyclic` with
# `anderson = 5`. The designs have Gaussian columns scaled to ‖Xⱼ‖² = n, s
# coefficients ±1 at random positions and a signal-to-noise ratio of 4, as in
# the lasso benchmark of ProximalOPT.jl. Prints passes, times and the relative
# suboptimality of the solution, then the times of 50-λ paths (`LassoPath`, λmax
# to λmax/100) and a comparison of memories K. Every case runs until it has at
# least half a second of runs, and at least 3 runs (5 for the random orders);
# the times are the minimum for the deterministic orders, and the passes and
# times the median for the random ones.
#
#     julia --project=benchmark benchmark/order_anderson_bench.jl

using CoordinateDescent, ProximalBase, LinearAlgebra, Random, Statistics, Printf
BLAS.set_num_threads(1)

function design(; n, p, s, ρ = 0.0, common = 0.0, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if ρ > 0
    c = sqrt(1 - ρ^2)
    for j in 2:p
      @views X[:, j] .= ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  common > 0 && (X .= sqrt(1 - common) .* X .+ sqrt(common) .* randn(rng, n))
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  X, signal .+ sqrt(sum(abs2, signal) / n / 4) .* randn(rng, n)
end

const DESIGNS = [
  ("A: n=500 p=2500, independent",      (n = 500,  p = 2500,  s = 50)),
  ("B: n=500 p=2500, AR(1) 0.8",        (n = 500,  p = 2500,  s = 50, ρ = 0.8)),
  ("B': n=500 p=2500, AR(1) 0.95",      (n = 500,  p = 2500,  s = 50, ρ = 0.95)),
  ("C: n=5000 p=500, AR(1) 0.5",        (n = 5000, p = 500,   s = 50, ρ = 0.5)),
  ("D: n=200 p=10000, independent",     (n = 200,  p = 10000, s = 20)),
  ("E: n=500 p=2500, equicorrelated 0.5", (n = 500, p = 2500, s = 50, common = 0.5)),
  ("E': n=500 p=2500, equicorrelated 0.9", (n = 500, p = 2500, s = 50, common = 0.9)),
]

const SOLVERS = [          # name, options, random order
  ("cyclic",   CDOptions(; order = :cyclic),              false),
  ("random",   CDOptions(; order = :random),              true),
  ("active",   CDOptions(; order = :active),              true),
  ("Anderson", CDOptions(; order = :cyclic, anderson = 5), false),
]

objective(X, y, β, λ) = sum(abs2, y - X * β) / (2 * size(X, 1)) + λ * sum(abs, β)

function solve(X, y, λ, opts)
  x = SparseIterate(size(X, 2))
  info = CDInfo()
  t = @elapsed coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), opts, info)
  x, info.iterations, t
end

med(v) = sort(v)[(length(v) + 1) ÷ 2]

# runs f() until there are at least half a second of runs and at least 3 runs
# (5 with random), after one run to compile; returns the results
function repeat_runs(f, random)
  f()
  runs, total = [], 0.0
  while length(runs) < (random ? 5 : 3) || total < 0.5
    t = @elapsed r = f()
    push!(runs, (r, t))
    total += t
  end
  runs
end

# passes and time (median with a random order, otherwise the minimum time), and
# the solution of the last run
function measure(X, y, λ, opts, random)
  runs = repeat_runs(() -> solve(X, y, λ, opts), random)
  passes = [r[1][2] for r in runs]
  ts = [r[2] for r in runs]
  random ? (med(passes), med(ts), runs[end][1][1]) : (passes[argmin(ts)], minimum(ts), runs[end][1][1])
end

println("# One λ, from β = 0, optTol = 1e-7: passes, milliseconds and (F - F*)/F*\n")
println("| design | λ/λmax | cyclic | random | active | Anderson |")
println("|---|---:|---:|---:|---:|---:|")
for (name, kw) in DESIGNS
  X, y = design(; kw...)
  n = size(X, 1)
  λmax = maximum(abs, X' * y) / n
  for r in (0.1, 0.02, 0.01)
    λ = r * λmax
    xref, _, _ = solve(X, y, λ, CDOptions(; order = :random, optTol = 1e-13, maxIter = 10^7))
    Fstar = objective(X, y, Vector(xref), λ)
    cells = String[]
    for (_, opts, random) in SOLVERS
      passes, t, x = measure(X, y, λ, opts, random)
      gap = max(objective(X, y, Vector(x), λ) - Fstar, 0) / Fstar
      push!(cells, @sprintf("%d, %.1f ms (%.0e)", passes, 1e3t, gap))
    end
    println("| ", name, " | ", r, " | ", join(cells, " | "), " |")
  end
end

println("\n# 50-λ path from λmax to λmax/100 (LassoPath, standardizeX = false): milliseconds\n")
println("| design | cyclic | random | active | Anderson |")
println("|---|---:|---:|---:|---:|")
for (name, kw) in DESIGNS[[1, 2, 5, 6]]
  X, y = design(; kw...)
  cells = String[]
  for (_, opts, random) in SOLVERS
    ts = [r[2] for r in repeat_runs(() -> LassoPath(X, y, opts; nλ = 50, λminratio = 0.01, standardizeX = false), random)]
    push!(cells, @sprintf("%.0f", 1e3 * (random ? med(ts) : minimum(ts))))
  end
  println("| ", name, " | ", join(cells, " | "), " |")
end

println("\n# Memory K of Anderson acceleration at λ = 0.02 λmax: passes, milliseconds\n")
println("| design | K = 3 | K = 5 | K = 10 |")
println("|---|---:|---:|---:|")
for (name, kw) in DESIGNS[[1, 2, 5]]
  X, y = design(; kw...)
  λ = 0.02 * maximum(abs, X' * y) / size(X, 1)
  cells = String[]
  for K in (3, 5, 10)
    passes, t, _ = measure(X, y, λ, CDOptions(; order = :cyclic, anderson = K), false)
    push!(cells, @sprintf("%d, %.1f ms", passes, 1e3t))
  end
  println("| ", name, " | ", join(cells, " | "), " |")
end
