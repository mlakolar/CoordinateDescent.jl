# Lasso paths, refitting, and choosing λ
#
# LassoPath solves the lasso
#
#     minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ⱼ ωⱼ |βⱼ|
#
# along a decreasing sequence of λ, each from the previous solution;
# refit_lasso_path refits least squares on the selected variables. This example
# shows the options of both and chooses λ by cross-validation.
#
#     julia --project=examples examples/03_lasso_path.jl

using CoordinateDescent
using ProximalBase: MCP
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(3)

# more predictors than observations: n = 100, p = 500, ten nonzero
# coefficients between 0.5 and 2 in absolute value, and an intercept
n, p, s = 100, 500, 10
X = randn(n, p)
β = zeros(p)
β[1:s] = range(2, 0.5; length = s) .* rand((-1, 1), s)
y = 1.0 .+ X * β + randn(n)

nonzeros(b) = count(!iszero, b)

println("== The default path")

# 100 values of λ, evenly spaced on the log scale from λmax, the smallest λ
# with a zero solution, down to λminratio ⋅ λmax, with λminratio = 0.01 when
# p > n and 1e-4 otherwise; loadings ωⱼ = standard deviation of X[:, j]
# (standardize_x = true); no intercept unless asked for; and screening with the
# sequential strong rule
path = LassoPath(X, y; intercept = true)
println("        λ  nonzeros    β̂₀  passes  KKT checks")
for k in 1:11:length(path.λpath)
  @printf("  %7.4f  %8d  %5.2f  %6d  %10d\n", path.λpath[k], nonzeros(path.βpath[k]),
          path.β0path[k], path.iterations[k], path.kkt_checks[k])
end
println("all converged: ", all(path.converged))

# path.βpath[k] is a SparseVector, so a path takes memory in proportion to the
# number of nonzero coefficients
println("βpath[50] is a ", typeof(path.βpath[50]), " with ", nonzeros(path.βpath[50]), " nonzeros")

println("\n== Other grids")

# fewer values on a shorter range ...
short = LassoPath(X, y; nλ = 20, λminratio = 0.1, intercept = true)
@printf("nλ = 20, λminratio = 0.1: λ from %.3f down to %.3f\n", first(short.λpath), last(short.λpath))

# ... a grid of your own, which should decrease ...
λs = exp.(range(log(1.0), log(0.05); length = 30))
own = LassoPath(X, y, λs; intercept = true)
@printf("own grid: %d values, %d nonzeros at λ = %.2f\n", length(own.λpath),
        nonzeros(own.βpath[end]), own.λpath[end])

# ... or a path that stops after the first solution with more than max_hat_s
# nonzero coefficients; λpath then holds only the values solved for
early = LassoPath(X, y; intercept = true, max_hat_s = 20)
@printf("max_hat_s = 20: stops after %d of 100 values, with %d nonzeros\n",
        length(early.λpath), nonzeros(early.βpath[end]))

println("\n== Unstandardized columns")

# With standardize_x = false the loadings are 1, and columns with larger
# standard deviations are penalized less (see 01_lasso.jl). Here the columns
# have standard deviations mostly between 0.1 and 10, and the variables that
# enter first are those with large columns rather than those that matter.
Xs = X .* exp.(1.2 * randn(1, p))
function entry_order(path)            # variables in the order they become nonzero
  order = Int[]
  for b in path.βpath, j in findall(!iszero, b)
    j in order || push!(order, j)
  end
  order
end
for standardize_x in (true, false)
  path_s = LassoPath(Xs, y .- 1.0; standardize_x)
  println("standardize_x = ", rpad(standardize_x, 5), ": first ten variables to enter ",
          first(entry_order(path_s), 10))
end

println("\n== Screening and solver options")

# With screening = true (the default), each λ is solved first over the
# variables kept by the strong rule, and the optimality conditions of the
# others are checked afterwards; violators are added and the problem solved
# again. kkt_checks[k] counts the checks, 1 when the rule missed nothing.
# screening = false visits all p variables in every full pass. The solutions
# are the same up to the accuracy of the solver, which is lowest at the end of
# the path, where almost n variables are selected. Screening saves time in the
# full passes when most of many variables stay at zero, as with n = 100 and
# p = 5000 here.
Xw = randn(100, 5000)
yw = Xw[:, 1:s] * β[1:s] + randn(100)
time_path(; kwargs...) = minimum(@elapsed(LassoPath(Xw, yw; kwargs...)) for _ in 1:3)
p1 = LassoPath(Xw, yw)
p0 = LassoPath(Xw, yw; screening = false)
Δ = [norm(p1.βpath[k] - p0.βpath[k], Inf) for k in eachindex(p1.λpath)]
@printf("largest difference between the two paths: %.1e over the first 80 values of λ, %.1e over the last 20\n",
        maximum(Δ[1:80]), maximum(Δ[81:end]))
@printf("time: %.0f ms with screening, %.0f ms without\n", 1000time_path(), 1000time_path(; screening = false))
println("values of λ at which the strong rule missed a variable: ", count(>(1), p1.kkt_checks))

# Options of coordinate descent go in the last positional argument, here a
# tighter tolerance; screening needs warm starts, so it is off with
# warm_start = false
tight = LassoPath(Xw, yw, CDOptions(; opt_tol = 1e-9))
println("with opt_tol = 1e-9: ", sum(tight.iterations), " passes along the path instead of ", sum(p1.iterations))

println("\n== Choosing λ by cross-validation")

# K-fold cross-validation on the λ grid of the full data: the path is fit
# without one fold, and its predictions are scored on that fold
function cvlasso(X, y, λpath; K = 5, kwargs...)
  n = size(X, 1)
  fold = shuffle!(mod1.(1:n, K))
  mse = zeros(length(λpath), K)
  for k in 1:K
    train, test = fold .!= k, fold .== k
    pathk = fit(LassoPath, X[train, :], y[train], λpath; intercept = true, kwargs...)
    mse[:, k] = vec(mean(abs2, y[test] .- predict(pathk, X[test, :]); dims = 1))
  end
  vec(mean(mse; dims = 2)), vec(std(mse; dims = 2)) / sqrt(K)
end

cvm, cvse = cvlasso(X, y, path.λpath)
imin = argmin(cvm)
i1se = findfirst(cvm .<= cvm[imin] + cvse[imin])     # largest λ within one standard error

println("                       λ  nonzeros  CV error  ‖β̂ - β‖")
for (name, i) in (("minimum CV error", imin), ("one standard error", i1se))
  @printf("  %-18s  %6.4f  %8d  %8.3f  %8.3f\n", name, path.λpath[i], nonzeros(path.βpath[i]),
          cvm[i], norm(path.βpath[i] - β))
end

println("\n== The elastic net and MCP")

# LassoPath(X, y; alpha) is glmnet's elastic-net path, whose λmax is that of
# the lasso divided by α, and LassoPath(X, y; penalty = MCP(1, 3)) the path of
# ncvreg, each solution the local one reached from the previous. MCP stops
# shrinking coefficients larger than aλ, so its estimates at the same λ are less
# biased; cvlasso works on any path.
for (name, kw) in (("elastic net, α = 0.5", (alpha = 0.5,)), ("MCP, a = 3", (penalty = MCP(1, 3),)))
  pk = LassoPath(X, y; intercept = true, kw...)
  ck, _ = cvlasso(X, y, pk.λpath; kw...)
  i = argmin(ck)
  @printf("  %-20s  λmax %6.3f  at the minimum CV error: λ = %6.4f, %3d nonzeros, CV error %6.3f, ‖β̂ - β‖ = %.3f\n",
          name, pk.λpath[1], pk.λpath[i], nonzeros(pk.βpath[i]), ck[i], norm(pk.βpath[i] - β))
end

println("\n== Least squares refits")

# refit_lasso_path maps every distinct support S along the path to the least
# squares fit on X[:, S], with the intercept first when the path has one.
# Refitting removes the shrinkage of the lasso from the selected coefficients.
refits = refit_lasso_path(path, X, y)
println(length(refits), " distinct supports along the path")

S = findall(!iszero, path.βpath[i1se])
βS = refits[S]
βrefit = zeros(p)
βrefit[S] = βS[2:end]
@printf("on the %d variables selected with one standard error: ‖β̂ - β‖ = %.3f for the lasso, %.3f refit\n",
        length(S), norm(path.βpath[i1se] - β), norm(βrefit - β))
println("true      ", round.(β[1:s]; digits = 2))
println("lasso     ", round.(Vector(path.βpath[i1se][1:s]); digits = 2))
println("refit     ", round.(βrefit[1:s]; digits = 2))
