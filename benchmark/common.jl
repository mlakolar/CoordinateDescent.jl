# Designs and interleaved timing for the scripts of this folder.

using CoordinateDescent, ProximalBase, LinearAlgebra, Random, Statistics, Printf
BLAS.set_num_threads(1)

# Gaussian columns with AR(1) correlation ρ between neighbors, or correlation
# `common` between every pair, or `factors` common factors with Gaussian
# loadings; scaled to ‖Xⱼ‖² = n; s coefficients ±1 and a signal-to-noise ratio
# of 4
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

# the time of a solver: the median with a random order, otherwise the minimum
summarize(ts, random) = random ? med(ts) : minimum(ts)
