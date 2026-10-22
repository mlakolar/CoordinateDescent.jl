# Options of coordinate descent: CDOptions and CDInfo
#
# Every solver takes a CDOptions: the high-level ones (lasso, sqrtLasso,
# LassoPath) as their last positional argument, coordinateDescent! as its
# fourth. coordinateDescent! reports how the run went in a CDInfo, its fifth
# argument.
#
#     julia --project=examples examples/02_solver_options.jl

using CoordinateDescent, ProximalBase
using LinearAlgebra, Printf, Random

Random.seed!(2)

# n observations of p columns with correlation ρ between every pair (a common
# factor), and ten nonzero coefficients
function simulate(n, p, ρ)
  X = sqrt(ρ) * randn(n) .+ sqrt(1 - ρ) * randn(n, p)
  β = zeros(p)
  β[1:10:100] .= 1
  X, X * β + randn(n)
end

objective(X, y, λ, x) = sum(abs2, y - X * x) / (2 * size(X, 1)) + λ * norm(x, 1)

# solves the lasso with coordinateDescent!, from zero; returns the solution,
# the CDInfo and the shortest time in seconds of three runs
function solve(X, y, λ, options)
  x, info, t = SparseIterate(size(X, 2)), CDInfo(), Inf
  for _ in 1:3
    x, info = SparseIterate(size(X, 2)), CDInfo()
    t = min(t, @elapsed coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), options, info))
  end
  x, info, t
end

X, y = simulate(500, 200, 0.5)
λ = 0.05

println("== The defaults")
options = CDOptions()
for name in fieldnames(CDOptions)
  println("  ", rpad(name, 12), getfield(options, name))
end

println("\n== optTol: when to stop")

# Coordinate descent stops after a full pass and a pass over the active set
# in which no coefficient changes by more than the tolerance. The default 1e-7
# gives about seven correct digits of the coefficients.
xstar, = solve(X, y, λ, CDOptions(; optTol = 1e-14))
Fstar = objective(X, y, λ, xstar)
println("  optTol  passes  |F - F*| / F*  max |β̂ - β*|")
for optTol in (1e-2, 1e-4, 1e-7, 1e-10)
  x, info = solve(X, y, λ, CDOptions(; optTol))
  @printf("%8.0e  %6d  %13.1e  %12.1e\n", optTol, info.iterations,
          abs(objective(X, y, λ, x) - Fstar) / Fstar, norm(x - xstar, Inf))
end

println("\n== relativeTol: a tolerance that does not depend on the units")

# With relativeTol = true (the default) the change of βₖ is measured as
# |Δβₖ| ‖X[:, k]‖ / √n, the size of the change in the fitted values, and
# compared with optTol ‖y‖ / √n. Expressing y in a unit a thousand times
# smaller scales β̂ and λ by 1000 but changes neither the passes nor the
# accuracy. With relativeTol = false, |Δβₖ| is compared with optTol itself,
# so the same optTol gives about seven correct digits in one unit and ten in
# the other. (The random number generator is reset before each run, so that
# the coordinates are visited in the same order.)
println("  relativeTol  unit of y  passes  max |β̂ - β*| / max |β*|")
for relativeTol in (true, false), scale in (1, 1000)
  Random.seed!(10)
  x, info = solve(X, scale * y, scale * λ, CDOptions(; relativeTol))
  @printf("  %-11s  %9s  %6d  %10.1e\n", relativeTol, scale == 1 ? "1" : "1/1000",
          info.iterations, norm(x / scale - xstar, Inf) / norm(xstar, Inf))
end

println("\n== order and anderson: the order of the coordinates")

# order = :random (the default) visits the coordinates in a new random order
# in every pass, drawn from the global random number generator (seed it with
# Random.seed! to reproduce a run). order = :cyclic visits them in the order
# 1, 2, …, p, which is faster on independent columns but much slower when the
# columns share a common factor. order = :active shuffles only the passes over
# the active set; the full passes, which mostly check that the zero
# coefficients stay at zero, go in the order 1, 2, …, p. The older option
# randomize = true or false means order = :random or :cyclic.
#
# With order = :cyclic, anderson = K extrapolates from the last K + 1 passes
# over an unchanged active set (Anderson acceleration), and keeps the result
# when it lowers the objective. It saves passes where coordinate descent needs
# many, as at small penalties with correlated columns, but it keeps the
# weakness of the cyclic order under a common factor.
Xiid, yiid = simulate(500, 200, 0.0)
Xar = randn(200, 1000)
for j in 2:1000
  @views Xar[:, j] .= 0.8 .* Xar[:, j-1] .+ 0.6 .* Xar[:, j]
end
yar = Xar[:, 1:10:100] * ones(10) + randn(200)
λar = 0.02 * maximum(abs, Xar' * yar) / 200
println("  design               options               passes  time (ms)")
for (name, Xd, yd, λd) in (("independent columns", Xiid, yiid, λ), ("common factor", X, y, λ),
                           ("AR(1) 0.8, p > n", Xar, yar, λar)),
    (oname, opts) in (("order = :random", CDOptions()), ("order = :cyclic", CDOptions(; order = :cyclic)),
                      ("order = :active", CDOptions(; order = :active)),
                      ("cyclic, anderson = 5", CDOptions(; order = :cyclic, anderson = 5)))
  _, info, t = solve(Xd, yd, λd, opts)
  @printf("  %-19s  %-20s  %6d  %9.2f\n", name, oname, info.iterations, 1000t)
end

println("\n== warmStart and numSteps: where to start")

# warmStart = true (the default) starts from the x passed in. With
# warmStart = false, x is set to zero and the problem is solved along numSteps
# penalty levels, from λmax, the smallest λ with a zero solution, down to λ,
# each started from the previous solution and screened with the sequential
# strong rule. Every level is solved to the tolerance, so for one λ the path
# costs more passes than a direct start, unless a direct start converges
# slowly, as with many more columns than rows and a small λ; a short path then
# pays off.
Xwide, ywide = simulate(100, 5000, 0.0)
println("  design        λ / λmax  warmStart  numSteps  passes  time (ms)")
for (name, Xd, yd) in (("n=500, p=200", X, y), ("n=100, p=5000", Xwide, ywide))
  λmax = maximum(abs, Xd' * yd) / size(Xd, 1)
  for (warmStart, numSteps) in ((true, 50), (false, 10), (false, 50))
    _, info, t = solve(Xd, yd, 0.02λmax, CDOptions(; warmStart, numSteps))
    @printf("  %-13s  %8.2f  %-9s  %8s  %6d  %9.2f\n", name, 0.02, warmStart,
            warmStart ? "-" : string(numSteps), info.iterations, 1000t)
  end
end

# A warm start from a nearby solution is cheap: this solves along ten values
# of λ, each from the previous solution (LassoPath in 03_lasso_path.jl does
# this for you)
x = SparseIterate(size(X, 2))
f = CDLeastSquaresLoss(y, X)
info = CDInfo()
println("\n       λ  passes  nonzeros")
for λk in exp.(range(log(0.5), log(0.002); length = 10))
  coordinateDescent!(x, f, ProxL1(λk), CDOptions(), info)
  @printf("  %6.3f  %6d  %8d\n", λk, info.iterations, count(!iszero, x))
end

println("\n== maxIter and CDInfo")

# maxIter caps the number of passes. Running out of passes gives a warning
# and info.converged = false, and info.maxChange > info.tolerance tells how far
# the last pass was from the stopping rule (both scaled as with relativeTol).
flush(stdout)
x = SparseIterate(size(X, 2))
info = CDInfo()
coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), CDOptions(; maxIter = 5), info)
println("  converged = $(info.converged) after $(info.iterations) passes; ",
        @sprintf("maxChange = %.2e, tolerance = %.2e", info.maxChange, info.tolerance))

println("\n== Float32")

# The iterate, the data and the penalty share one floating point type. A
# Float32 solution is accurate to about 1e-7 relative to its size, so a
# tolerance at that level may never be met: use optTol ≥ 1e-6.
x32 = SparseIterate(Float32, size(X, 2))
info = CDInfo()
coordinateDescent!(x32, CDLeastSquaresLoss(Float32.(y), Float32.(X)), ProxL1(Float32(λ)),
                   CDOptions(; optTol = 1e-6), info)
@printf("  %s solution in %d passes, max |β̂ - β*| = %.1e\n", eltype(x32), info.iterations,
        norm(Vector(x32) - xstar, Inf))
