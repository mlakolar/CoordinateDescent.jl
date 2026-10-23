# The default options, which use momentum with the random order, against
# `momentum = false`, the default before version 0.6, for the high-level
# solvers and for CDQuadraticLoss, on designs A, E and F of common.jl and a
# varying-coefficient model. Each case runs the two in turn until each has at
# least half a second of runs and at least 5 runs, and prints the median times.
#
#     julia --project=benchmark benchmark/momentum_default_bench.jl

include("common.jl")

const SETTINGS = [("momentum = false", CDOptions(; momentum = false), true),
                  ("default",          CDOptions(),                   true)]

function row(label, run)
  _, times = interleaved(run, SETTINGS)
  t = [summarize(times[j], true) for j in eachindex(SETTINGS)]
  @printf("| %s | %.1f | %.1f | %.2f |\n", label, 1e3 * t[1], 1e3 * t[2], t[1] / t[2])
end

println("| case | momentum = false (ms) | default (ms) | speedup |")
println("|---|---:|---:|---:|")
for d in (1, 6, 8)
  name, kw = DESIGNS[d]
  X, y = design(; kw...)
  n, p = size(X)
  tag = split(name, ":")[1]
  λmax = maximum(abs, X' * y) / n
  row("$tag: `lasso`, 0.05 λmax, intercept, standardized",
      o -> lasso(X, y, 0.05 * λmax, o; intercept = true, standardizeX = true))
  row("$tag: `sqrtLasso`, λ = 1.1 √(2 log p / n)",
      o -> sqrtLasso(X, y, 1.1 * sqrt(2 * log(p) / n), o))
  row("$tag: `scaledLasso`, λ = √(2 log p / n)",
      o -> scaledLasso(X, y, sqrt(2 * log(p) / n), ones(p), IterLassoOptions(; optionsCD = o)))
  row("$tag: `feasibleLasso!`, λ0 = 2 √(2 log p / n)",
      o -> feasibleLasso!(SparseIterate(p), X, y, 2 * sqrt(2 * log(p) / n), IterLassoOptions(; optionsCD = o)))
  row("$tag: `LassoPath`, 100 λ, intercept, standardized",
      o -> LassoPath(X, y, o; intercept = true))
  G, b = X' * X ./ n, -(X' * y) ./ n
  row("$tag: `CDQuadraticLoss(X'X/n, -X'y/n)`, 0.02 λmax",
      o -> coordinateDescent!(SparseIterate(p), CDQuadraticLoss(G, b), ProxL1(0.02 * λmax), o))
end

# a varying-coefficient model, as in examples/07_varying_coefficient.jl
rng = Xoshiro(7)
n, p = 500, 50
z = rand(rng, n)
X = randn(rng, n, p)
y = X[:, 1] .* sin.(2π * z) + X[:, 2] .* 4z .* (1 .- z) + X[:, 3] + 0.3 * randn(rng, n)
row("`locpolyl1`, n = 500, p = 50, degree 1, 5 grid points",
    o -> locpolyl1(X, z, y, collect(0.1:0.2:0.9), 1, EpanechnikovKernel(0.2), 0.05, true, o))
