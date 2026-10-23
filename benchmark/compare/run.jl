# The cases of momentum_default_bench.jl, run by the version of
# CoordinateDescent in the active environment: old/ (0.6.0, on ProximalBase
# 0.4, with its default momentum and with momentum = false) or new/ (this
# checkout, on ProximalSolvers). Each setting of a case runs in turn with the
# others, after one run to compile, until each has half a second of runs and at
# least 5; every run starts from the same seed. Writes the median times, the
# passes and the solutions to results/<side>_<stamp>.jls; compare.jl compares
# the runs of the two sides. See benchmark/README.md.
#
#     julia --project=benchmark/compare/old benchmark/compare/run.jl old
#     julia --project=benchmark/compare/new benchmark/compare/run.jl new

using CoordinateDescent, ProximalBase, LinearAlgebra, Printf, Random, Serialization, SparseArrays, Statistics
BLAS.set_num_threads(1)

const SIDE = ARGS[1]
const OLD = SIDE == "old"
include(joinpath(@__DIR__, "..", "common.jl"))

# the ℓ1 penalty in both versions of ProximalBase
l1(λ) = isdefined(ProximalBase, :NormL1) ? ProximalBase.NormL1(λ) : ProximalBase.ProxL1(λ)

const SETTINGS = OLD ? [("0.6", CDOptions()), ("0.6, momentum = false", CDOptions(; momentum = false))] :
                       [("0.7", CDOptions())]

# a vector of the solution, and the passes
out(s) = (Vector(s.x), s.iterations)

# the cases on one design of common.jl, each in a function of its own data
function design_cases(d)
  name, kw = DESIGNS[d]
  X, y = design(; kw...)
  n, p = size(X)
  tag = split(name, ":")[1]
  λmax = maximum(abs, X' * y) / n
  G, b = X' * X ./ n, -(X' * y) ./ n
  Tuple{String, Function}[
    ("$tag lasso", o -> out(lasso(X, y, 0.05 * λmax, o; intercept = true, standardizeX = true))),
    ("$tag sqrtLasso", o -> out(sqrtLasso(X, y, 1.1 * sqrt(2 * log(p) / n), o))),
    ("$tag scaledLasso", o -> out(scaledLasso(X, y, sqrt(2 * log(p) / n), ones(p), IterLassoOptions(; optionsCD = o)))),
    ("$tag feasibleLasso!", o -> out(feasibleLasso!(SparseIterate(p), X, y, 2 * sqrt(2 * log(p) / n), IterLassoOptions(; optionsCD = o)))),
    ("$tag LassoPath", o -> (P = LassoPath(X, y, o; intercept = true); (reduce(vcat, Vector.(P.βpath)), sum(P.iterations)))),
    ("$tag CDQuadraticLoss", o -> (x = SparseIterate(p); info = CDInfo();
                                   coordinateDescent!(x, CDQuadraticLoss(G, b), l1(0.02 * λmax), o, info);
                                   (Vector(x), info.iterations)))]
end

# a varying-coefficient model, as in examples/07_varying_coefficient.jl
function locpoly_case()
  rng = Xoshiro(7)
  n, p = 500, 50
  z = rand(rng, n)
  X = randn(rng, n, p)
  y = X[:, 1] .* sin.(2π * z) + X[:, 2] .* 4z .* (1 .- z) + X[:, 3] + 0.3 * randn(rng, n)
  ("locpolyl1", o -> (B = locpolyl1(X, z, y, collect(0.1:0.2:0.9), 1, EpanechnikovKernel(0.2), 0.05, true, o);
                      (vec(Matrix(B[1])), 0)))
end

cases() = [design_cases(1); design_cases(6); design_cases(8); locpoly_case()]

results = Dict{Tuple{String, String}, Any}()
for (case, run) in cases()
  runs = [(o -> (Random.seed!(20261023); run(o)), o) for (_, o) in SETTINGS]
  for (r, o) in runs
    r(o)
  end
  times = [Float64[] for _ in SETTINGS]
  sols = Vector{Any}(undef, length(SETTINGS))
  enough(j) = length(times[j]) >= 5 && sum(times[j]) >= 0.5
  while !all(enough, eachindex(SETTINGS))
    for (j, (r, o)) in enumerate(runs)
      enough(j) && continue
      t = @elapsed sols[j] = r(o)
      push!(times[j], t)
    end
  end
  for (j, (label, _)) in enumerate(SETTINGS)
    results[(case, label)] = (time = med(times[j]), passes = sols[j][2], solution = sols[j][1])
    @printf("%-22s %-24s %9.2f ms %8d passes\n", case, label, 1e3 * med(times[j]), sols[j][2])
  end
end
mkpath(joinpath(@__DIR__, "results"))
file = joinpath(@__DIR__, "results", "$(SIDE)_$(round(Int, time())).jls")
serialize(file, results)
println("wrote ", file)
