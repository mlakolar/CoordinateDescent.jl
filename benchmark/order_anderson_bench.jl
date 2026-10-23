# Coordinate orders, Anderson acceleration (issue #9) and momentum (issue #20)
# on the lasso
#
#     minimize  ‖y - Xβ‖² / (2n) + λ‖β‖₁,
#
# solved from β = 0 with the default stopping rule (optTol = 1e-7), for
# `order = :cyclic` and `:random` without momentum, `order = :cyclic` with
# `anderson = 5`, and `order = :random` with `momentum = true`. Version 0.6
# also had `order = :active`, which version 0.7 removed; see README.md. The
# designs are those of common.jl, as in the lasso benchmark of ProximalOPT.jl.
# Prints passes, times and the relative suboptimality of the solution, then the
# times of 50-λ paths (`LassoPath`, λmax to λmax/100) and a comparison of
# memories K.
# Other processes load the machine unevenly, so every case runs the solvers in
# turn, until each has at least half a second of runs and at least 3 runs (5 for
# the random orders); the times are the minimum for the deterministic orders,
# and the passes and times the median for the random ones.
#
#     julia --project=benchmark benchmark/order_anderson_bench.jl

include("common.jl")

const SOLVERS = [          # name, options, random order
  ("cyclic",   CDOptions(; order = :cyclic),                  false),
  ("random",   CDOptions(; order = :random, momentum = false), true),
  ("Anderson", CDOptions(; order = :cyclic, anderson = 5),     false),
  ("momentum", CDOptions(; order = :random, momentum = true),  true),
]

objective(X, y, β, λ) = sum(abs2, y - X * β) / (2 * size(X, 1)) + λ * sum(abs, β)

function solve(X, y, λ, opts)
  x = SparseIterate(size(X, 2))
  info = CDInfo()
  coordinateDescent!(x, CDLeastSquaresLoss(y, X), NormL1(λ), opts, info)
  x, info.iterations
end


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
