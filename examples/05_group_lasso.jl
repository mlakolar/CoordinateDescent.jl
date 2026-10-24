# The group lasso
#
#     minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ₖ wₖ ‖β[Gₖ]‖₂
#
# by block coordinate descent: each step updates one group Gₖ of
# coefficients, and the iterate is an AtomIterate that knows the groups.
# Groups arise when a predictor is coded by several columns, such as the dummy
# columns of a categorical variable or a basis expansion of a nonlinear
# effect: the group lasso keeps or drops the columns of a predictor together.
#
#     julia --project=examples examples/05_group_lasso.jl

using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(5)

# An additive model y = β₀ + f₁(z₁) + f₂(z₂) + f₃(z₃) + ε in J = 30 variables
# zⱼ ~ Uniform(-1, 1), of which three matter. Each variable enters through a
# cubic polynomial, three standardized columns zⱼ, zⱼ², zⱼ³ that form a group.
n, J = 300, 30
Z = 2 * rand(n, J) .- 1
standardize(v) = (v .- mean(v)) / std(v)
X = reduce(hcat, [standardize(Z[:, j] .^ d) for j in 1:J for d in 1:3])
p = size(X, 2)
groups = [3(j-1)+1:3j for j in 1:J]
y = 1.0 .+ sin.(π * Z[:, 1]) + 2 * Z[:, 2] .^ 2 + Z[:, 3] + 0.5 * randn(n)

selected(x) = findall(G -> any(!iszero, x[G]), groups)

println("== One λ")

# GroupNormL2(λ, groups) takes the weights wₖ = √|Gₖ|, AtomIterate(p, groups)
# is a zero iterate with these groups, and the intercept is not penalized. The
# groups must cover all coefficients.
λ = 0.05
g = GroupNormL2(λ, groups)
f = CDLeastSquaresLoss(y, X; intercept = true)
x = AtomIterate(p, groups)
info = CDInfo()
coordinateDescent!(x, f, g, CDOptions(), info)
β₀ = ProximalSolvers.intercept(f, x)              # the intercept ȳ - x̄'β̂
println("selected variables ", selected(x), " in $(info.iterations) passes; β̂₀ = ", round(β₀; digits = 2))
println("norms of the groups ", round.([norm(x[G]) for G in groups[1:6]]; digits = 2), " …")

# A solution satisfies ‖∇_G f‖ ≤ λ wₖ for the groups at zero and
# ∇_G f = -λ wₖ β̂_G / ‖β̂_G‖ for the others, with ∇f = -(X - 1x̄')'r / n
r = y .- β₀ - X * x
∇f = -(X .- mean(X; dims = 1))' * r / n
kkt = maximum(enumerate(groups)) do (k, G)
  w = g.w[k]
  iszero(norm(x[G])) ? max(norm(∇f[G]) - λ * w, 0.0) : norm(∇f[G] + λ * w * x[G] / norm(x[G]))
end
@printf("largest violation of the optimality conditions: %.1e\n", kkt)

println("\n== Group weights")

# GroupNormL2(λ, groups, w) takes weights of your own. A zero weight leaves a
# group unpenalized, so that it is always in the model, here variable 10.
w = sqrt.(length.(groups))
w[10] = 0
gw = GroupNormL2(λ, groups, w)
xw = AtomIterate(p, groups)
coordinateDescent!(xw, f, gw)
println("with w₁₀ = 0: selected variables ", selected(xw))

println("\n== Starting from zero along a path of penalties")

# With warmStart = false, coordinate descent solves along numSteps penalty
# levels from λmax = maxₖ ‖∇_Gₖ f(0)‖ / wₖ down to λ, screening the groups with
# the sequential strong rule at each level. As for the lasso
# (02_solver_options.jl), this costs more passes than a direct start unless a
# direct start converges slowly.
λsmall = 0.01
for (warmStart, numSteps) in ((true, 50), (false, 10), (false, 50))
  xs = AtomIterate(p, groups)
  coordinateDescent!(xs, f, GroupNormL2(λsmall, groups), CDOptions(; warmStart, numSteps), info)
  @printf("warmStart = %-5s numSteps = %-3s: %4d passes, %d groups selected\n", warmStart,
          warmStart ? "-" : string(numSteps), info.iterations, length(selected(xs)))
end

println("\n== Choosing λ on a validation set")

# A loop over decreasing λ starts each fit from the previous solution, as
# ProximalSolvers's solve_path does (with screening, and a callback for a
# validation error). Here λ is chosen by the error on 100 held-out
# observations.
train, valid = 1:200, 201:300
ft = CDLeastSquaresLoss(y[train], X[train, :]; intercept = true)
∇0 = X[train, :]' * (y[train] .- mean(y[train])) / length(train)
λmax = maximum(norm(∇0[G]) / sqrt(length(G)) for G in groups)
xt = AtomIterate(p, groups)
println("       λ  groups  validation error")
best = (Inf, 0.0)
for λk in λmax * exp10.(range(0, -2.5; length = 15))
  gk = GroupNormL2(λk, groups)
  coordinateDescent!(xt, ft, gk)                  # warm start from the previous λ
  pred = ProximalSolvers.intercept(ft, xt) .+ X[valid, :] * xt
  err = mean(abs2, y[valid] - pred)
  @printf("  %6.4f  %6d  %16.4f\n", λk, length(selected(xt)), err)
  err < best[1] && (global best = (err, λk))
end
println("smallest validation error at λ = ", round(best[2]; digits = 4))

println("\n== The quadratic loss, and a dense starting point")

# CDQuadraticLoss(A, b), f(x) = x'Ax/2 + b'x, with A = X'X/n and b = -X'y/n
# for centered X and y has the same minimizer; an update then costs O(p)
# instead of O(n), which pays off when n is much larger than p. A dense Vector
# is accepted as well; it is overwritten with the solution.
Xc = X .- mean(X; dims = 1)
yc = y .- mean(y)
xq = zeros(p)
coordinateDescent!(xq, CDQuadraticLoss(Xc' * Xc / n, -Xc' * yc / n), g, CDOptions(; optTol = 1e-10))
xl = AtomIterate(p, groups)
coordinateDescent!(xl, f, g, CDOptions(; optTol = 1e-10))
@printf("largest difference to the least squares loss: %.1e\n", norm(xq - Vector(xl), Inf))

println("\n== The estimators: groupLasso and GroupLassoPath")

# groupLasso(X, y, λ, groups) and GroupLassoPath(X, y, groups) fit an
# intercept and, by default, orthonormalize each group: the penalty
# λ ∑ₖ wₖ ‖X̃_Gₖ β_Gₖ‖₂ / √n on the fitted values of the group (X̃ centered)
# does not depend on the basis of the group, so rescaling a column of the
# polynomial basis changes only its own coefficient, and every block step is
# exact. standardize = :none penalizes λ ∑ₖ wₖ ‖β_Gₖ‖₂ as above.
sol = groupLasso(X, y, λ, groups)
println("orthonormal groups: selected variables ", sol.activegroups, " in $(sol.iterations) passes")
sol_none = groupLasso(X, y, λ, groups; standardize = :none)
println("columns as given:   selected variables ", sol_none.activegroups)
Xs = copy(X); Xs[:, 2] .*= 10
@printf("scaling column 2 by 10 divides its coefficient by %.2f\n", sol.x[2] / groupLasso(Xs, y, λ, groups).x[2])

# The path from λmax, with the group strong rule, and λ chosen on the validation set
path = GroupLassoPath(X[train, :], y[train], groups; nλ = 30)
errs = [mean(abs2, y[valid] .- path.β0path[k] .- X[valid, :] * path.βpath[k]) for k in eachindex(path.λpath)]
k = argmin(errs)
@printf("best λ = %.4f: variables %s, validation error %.4f\n", path.λpath[k], path.activegroups[k], errs[k])
