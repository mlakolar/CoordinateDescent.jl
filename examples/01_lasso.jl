# The lasso
#
#     minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ⱼ ωⱼ |βⱼ|
#
# with `lasso` and its parameters: the penalty level λ, an unpenalized
# intercept β₀, standardized columns, and penalty loadings ωⱼ.
#
#     julia --project=examples examples/01_lasso.jl

using CoordinateDescent
using ProximalBase: MCP, SCAD
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(1)

# n = 200 observations of p = 50 predictors, of which the first five matter
n, p = 200, 50
X = randn(n, p)
β = [3.0, -2.0, 1.5, -1.0, 0.5, zeros(p - 5)...]
y = X * β + randn(n)

support(x) = findall(!iszero, x)

println("== The penalty level λ")

# λ = σ √(2 log(p) / n) is the choice of the theory, with noise level σ = 1
# here; 03_lasso_path.jl chooses λ by cross-validation and
# 04_sqrt_scaled_feasible.jl without knowing σ.
λ = sqrt(2 * log(p) / n)
sol = lasso(X, y, λ)

@printf("λ = %.3f selects %s\n", λ, support(sol.x))
println("β̂[1:5] = ", round.(sol.x[1:5]; digits = 2), ", shrunk towards zero")
println("σ̂ = ", round(sol.σ; digits = 3), ", the standard deviation of the residuals")
println("$(sol.iterations) passes over the coordinates, converged: $(sol.converged)")

# a larger λ gives a sparser solution; all coefficients are zero for
# λ ≥ λmax = maxⱼ |X[:, j]'y| / n
λmax = maximum(abs, X' * y) / n
println("\n     λ  nonzeros  ‖β̂ - β‖")
for λ in (1.01λmax, 0.5λmax, 0.2, 0.1, 0.05, 0.01)
  sol = lasso(X, y, λ)
  @printf("%6.3f  %8d  %7.3f\n", λ, count(!iszero, sol.x), norm(sol.x - β))
end

println("\n== An intercept")

# With intercept = true, the intercept is not penalized and
# sol.intercept = ȳ - x̄'β̂. The columns and y are centered on the fly, so X is
# not copied. Without an intercept, the predictors have to explain the mean of
# y as well.
y2 = 2.0 .+ y
for intercept in (false, true)
  sol = lasso(X, y2, λ; intercept)
  @printf("intercept = %-5s  β̂₀ = %.2f  RMS of the residuals %.2f  selects %s\n",
          intercept, sol.intercept, sqrt(mean(abs2, residuals(sol))), support(sol.x))
end

# predictions at new points, and the other functions of StatsAPI: coef(sol) is
# [β̂₀; β̂] with an intercept, dof(sol) the number of nonzero coefficients plus
# one for the intercept
sol = lasso(X, y2, λ; intercept = true)
Xnew = randn(3, p)
ŷ = predict(sol, Xnew)                         # sol.intercept .+ Xnew * sol.x
println("predictions ", round.(ŷ; digits = 2), ", true means ", round.(2.0 .+ Xnew * β; digits = 2))
println("coef(sol)[1:3] = ", round.(Vector(coef(sol))[1:3]; digits = 2), ", nobs = ", nobs(sol), ", dof = ", dof(sol))

println("\n== Standardization")

# The penalty λ ∑ⱼ |βⱼ| depends on the units of the columns. Measuring
# predictor 10, which does not matter, in centimeters instead of meters
# multiplies its column by 100, so that a coefficient a hundred times smaller
# fits the same, at a hundredth of the penalty. standardizeX = true takes the
# loadings ωⱼ = standard deviation of X[:, j], which gives the lasso on
# standardized columns with the coefficients transformed back: the fit then
# does not depend on the units.
Xcm = copy(X)
Xcm[:, 10] .*= 100
for standardizeX in (false, true)
  a = lasso(X, y, λ; standardizeX)
  b = lasso(Xcm, y, λ; standardizeX)
  @printf("standardizeX = %-5s  selects %s in meters and %s in centimeters\n",
          standardizeX, support(a.x), support(b.x))
end

println("\n== Penalty loadings ωⱼ")

# lasso(X, y, λ, ω) penalizes λ ωⱼ |βⱼ|. A zero loading leaves a coefficient
# unpenalized, so that its variable is always in the model, for example a
# treatment whose effect is of interest.
ω = ones(p)
ω[[20, 30]] .= 0
sol = lasso(X, y, λ, ω)
println("with ω₂₀ = ω₃₀ = 0 the lasso selects ", support(sol.x))

# The adaptive lasso (Zou, 2006) takes ωⱼ = 1 / |β̃ⱼ| from a first estimate β̃,
# here least squares (n > p). Large coefficients are penalized less, so they
# are less biased, and small ones more, so fewer variables that do not matter
# are selected as λ decreases. adaptiveLasso(X, y, λ; init = β̃) is
# lasso(X, y, λ, 1 ./ abs.(β̃)).
β̃ = X \ y
println("\n           lasso               adaptive lasso")
println("     λ  nonzeros  ‖β̂ - β‖   nonzeros  ‖β̂ - β‖")
for λ in (0.2, 0.1, 0.05, 0.02)
  a = lasso(X, y, λ)
  b = adaptiveLasso(X, y, λ; init = β̃)
  @printf("%6.2f  %8d  %7.3f   %8d  %7.3f\n",
          λ, count(!iszero, a.x), norm(a.x - β), count(!iszero, b.x), norm(b.x - β))
end

# By default the first estimate is the lasso at the same λ (init = :lasso, or
# :ridge, at the level initλ): a variable that it drops stays out.
c = adaptiveLasso(X, y, 0.05)
println("from a first lasso at λ = 0.05, the adaptive lasso selects ", support(c.x))

println("\n== Other penalties")

# alpha < 1 gives the elastic net λ(α‖β‖₁ + (1 - α)‖β‖²/2) of glmnet, which
# keeps groups of correlated variables together; penalty gives MCP or SCAD,
# whose level λ replaces. With standardizeX the penalty applies to the
# coefficients of the standardized columns, as in glmnet and ncvreg. At the
# level λ of the theory, MCP and SCAD leave the coefficients larger than aλ
# unshrunk, so their estimates are closer to β.
for (name, kw) in (("lasso", (;)), ("elastic net, α = 0.5", (alpha = 0.5,)),
                   ("MCP, a = 3", (penalty = MCP(1, 3),)), ("SCAD, a = 3.7", (penalty = SCAD(1, 3.7),)))
  est = lasso(X, y, λ; intercept = true, standardizeX = true, kw...)
  @printf("%-22s selects %2d variables, ‖β̂ - β‖ = %.3f\n", name, count(!iszero, est.x), norm(est.x - β))
end

println("\n== Number types and solver options")

# lasso converts X, y and λ to a common floating point type: integer data
# give a Float64 solution and Float32 data a Float32 one
sol32 = lasso(Float32.(X), Float32.(y), 0.2)
println("eltype of the solution from Float32 data: ", eltype(sol32.x))

# The last positional argument sets the options of coordinate descent
# (02_solver_options.jl), here a tighter tolerance
sol = lasso(X, y, λ, CDOptions(; optTol = 1e-10))
println("with optTol = 1e-10: $(sol.iterations) passes")
