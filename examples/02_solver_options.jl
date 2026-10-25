# Options of coordinate descent: CDOptions, keywords, and what a run reports
#
# Every solver takes a CDOptions: the high-level ones (lasso, sqrt_lasso,
# LassoPath) as their last positional argument, coordinate_descent! as its
# fourth. They also take its fields as keywords, as in
# lasso(X, y, λ; opt_tol = 1e-8), and a random number generator rng for the
# random order. They run the coordinate descent of ProximalSolvers (CDSolver)
# with these options. solve!(x, f, g, options) does what coordinate_descent!
# does and returns how the run went (a CDResults).
#
#     julia --project=examples examples/02_solver_options.jl

using HighDimensionalModels, ProximalBase
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

# solves the lasso with solve!, from zero; returns the solution, the results
# (passes, converged, …) and the shortest time in seconds of three runs
function timed_solve(X, y, λ, options)
  x, r, t = SparseIterate(size(X, 2)), nothing, Inf
  for _ in 1:3
    x = SparseIterate(size(X, 2))
    t = min(t, @elapsed r = solve!(x, CDLeastSquaresLoss(y, X), NormL1(λ), options))
  end
  x, r, t
end

X, y = simulate(500, 200, 0.5)
λ = 0.05

println("== The defaults")
options = CDOptions()
for name in fieldnames(CDOptions)
  println("  ", rpad(name, 12), getfield(options, name))
end

println("\n== opt_tol: when to stop")

# Coordinate descent stops after a full pass and a pass over the active set
# in which no coefficient changes by more than the tolerance. The default 1e-7
# gives about seven correct digits of the coefficients.
xstar, = timed_solve(X, y, λ, CDOptions(; opt_tol = 1e-14))
Fstar = objective(X, y, λ, xstar)
println("  opt_tol  passes  |F - F*| / F*  max |β̂ - β*|")
for opt_tol in (1e-2, 1e-4, 1e-7, 1e-10)
  x, r = timed_solve(X, y, λ, CDOptions(; opt_tol))
  @printf("%8.0e  %6d  %13.1e  %12.1e\n", opt_tol, r.passes,
          abs(objective(X, y, λ, x) - Fstar) / Fstar, norm(x - xstar, Inf))
end

println("\n== relative_tol: a tolerance that does not depend on the units")

# With relative_tol = true (the default) the change of βₖ is measured as
# |Δβₖ| ‖X[:, k]‖ / √n, the size of the change in the fitted values, and
# compared with opt_tol ‖y‖ / √n. Expressing y in a unit a thousand times
# smaller scales β̂ and λ by 1000 but changes neither the passes nor the
# accuracy. With relative_tol = false, |Δβₖ| is compared with opt_tol itself,
# so the same opt_tol gives about seven correct digits in one unit and ten in
# the other. (The random number generator is reset before each run, so that
# the coordinates are visited in the same order.)
println("  relative_tol  unit of y  passes  max |β̂ - β*| / max |β*|")
for relative_tol in (true, false), scale in (1, 1000)
  Random.seed!(10)
  x, r = timed_solve(X, scale * y, scale * λ, CDOptions(; relative_tol))
  @printf("  %-11s  %9s  %6d  %10.1e\n", relative_tol, scale == 1 ? "1" : "1/1000",
          r.passes, norm(x / scale - xstar, Inf) / norm(xstar, Inf))
end

println("\n== order and anderson: the order of the coordinates")

# order = :random (the default) visits the coordinates in a new random order
# in every pass, drawn from the random number generator rng, a keyword of every
# solver (by default the global generator, which Random.seed! resets), and by
# default with momentum (see below).
# order = :cyclic visits them in the order 1, 2, …, p, which is faster on
# independent columns but much slower when the columns share a common factor.
# The older option randomize = true or false means order = :random or :cyclic.
#
# With order = :cyclic, anderson = K extrapolates from the last K + 1 passes
# over an unchanged active set (Anderson acceleration), and keeps the result
# when it lowers the objective. It saves passes where coordinate descent needs
# many, as at small penalties with correlated columns, but it keeps the
# weakness of the cyclic order under a common factor. With order = :random,
# each pass over the active set starts from an extrapolated point (Nesterov
# momentum), by default for the built-in losses; it works with the random
# order, so it also helps under a common factor. momentum = false turns it off.
Xiid, yiid = simulate(500, 200, 0.0)
Xar = randn(200, 1000)
for j in 2:1000
  @views Xar[:, j] .= 0.8 .* Xar[:, j-1] .+ 0.6 .* Xar[:, j]
end
yar = Xar[:, 1:10:100] * ones(10) + randn(200)
λar = 0.02 * maximum(abs, Xar' * yar) / 200
println("  design               options                   passes  time (ms)")
for (name, Xd, yd, λd) in (("independent columns", Xiid, yiid, λ), ("common factor", X, y, λ),
                           ("AR(1) 0.8, p > n", Xar, yar, λar)),
    (oname, opts) in (("order = :random", CDOptions()),
                      ("random, momentum = false", CDOptions(; momentum = false)),
                      ("order = :cyclic", CDOptions(; order = :cyclic)),
                      ("cyclic, anderson = 5", CDOptions(; order = :cyclic, anderson = 5)))
  _, r, t = timed_solve(Xd, yd, λd, opts)
  @printf("  %-19s  %-24s  %6d  %9.2f\n", name, oname, r.passes, 1000t)
end

println("\n== warm_start and num_steps: where to start")

# warm_start = true (the default) starts from the x passed in. With
# warm_start = false, x is set to zero and the problem is solved along num_steps
# penalty levels, from λmax, the smallest λ with a zero solution, down to λ,
# each started from the previous solution and screened with the sequential
# strong rule. Every level is solved to the tolerance, so for one λ the path
# costs more passes than a direct start, unless a direct start converges
# slowly, as with many more columns than rows and a small λ; a short path then
# pays off.
Xwide, ywide = simulate(100, 5000, 0.0)
println("  design        λ / λmax  warm_start  num_steps  passes  time (ms)")
for (name, Xd, yd) in (("n=500, p=200", X, y), ("n=100, p=5000", Xwide, ywide))
  λmax = maximum(abs, Xd' * yd) / size(Xd, 1)
  for (warm_start, num_steps) in ((true, 50), (false, 10), (false, 50))
    _, r, t = timed_solve(Xd, yd, 0.02λmax, CDOptions(; warm_start, num_steps))
    @printf("  %-13s  %8.2f  %-9s  %8s  %6d  %9.2f\n", name, 0.02, warm_start,
            warm_start ? "-" : string(num_steps), r.passes, 1000t)
  end
end

# A warm start from a nearby solution is cheap: this solves along ten values
# of λ, each from the previous solution (LassoPath in 03_lasso_path.jl does
# this for you)
x = SparseIterate(size(X, 2))
f = CDLeastSquaresLoss(y, X)
println("\n       λ  passes  nonzeros")
for λk in exp.(range(log(0.5), log(0.002); length = 10))
  r = solve!(x, f, NormL1(λk), CDOptions())
  @printf("  %6.3f  %6d  %8d\n", λk, r.passes, count(!iszero, x))
end

println("\n== max_iter, and what a run reports")

# max_iter caps the number of passes. Running out of passes gives a warning and
# r.converged = false, and r.maxchange > r.tolerance tells how far the last
# pass was from the stopping rule (both scaled as with relative_tol). The
# options can also be given as keywords, which change those passed.
flush(stdout)
x = SparseIterate(size(X, 2))
r = solve!(x, CDLeastSquaresLoss(y, X), NormL1(λ), CDOptions(); max_iter = 5)
println("  converged = $(r.converged) after $(r.passes) passes; ",
        @sprintf("maxchange = %.2e, tolerance = %.2e", r.maxchange, r.tolerance))

println("\n== rng: reproducible random orders")

# A generator of its own makes the random order, and so the passes, the same
# from run to run without reseeding the global generator, and gives parallel
# fits independent streams.
passes(rng) = lasso(X, y, λ; rng).iterations
println("  passes with Xoshiro(1), twice: ", passes(Xoshiro(1)), ", ", passes(Xoshiro(1)))

println("\n== Float32")

# The iterate, the data and the penalty share one floating point type. A
# Float32 solution is accurate to about 1e-7 relative to its size, so a
# tolerance at that level may never be met: use opt_tol ≥ 1e-6.
x32 = SparseIterate(Float32, size(X, 2))
r = solve!(x32, CDLeastSquaresLoss(Float32.(y), Float32.(X)), NormL1(Float32(λ)), CDOptions(; opt_tol = 1e-6))
@printf("  %s solution in %d passes, max |β̂ - β*| = %.1e\n", eltype(x32), r.passes,
        norm(Vector(x32) - xstar, Inf))
