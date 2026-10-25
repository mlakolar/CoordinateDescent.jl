# Choosing the penalty without knowing the noise level
#
# The penalty of the lasso should be proportional to the noise level σ, which
# is rarely known. Three estimators get around that:
#
# - sqrt_lasso, the square-root lasso of Belloni, Chernozhukov and Wang (2011),
#       minimize  ‖y - β₀ - Xβ‖ + λ ∑ⱼ ωⱼ |βⱼ|,
#   whose λ does not depend on σ;
# - scaled_lasso, the scaled lasso of Sun and Zhang (2012), which estimates σ
#   jointly with β and penalizes λ σ̂ ∑ⱼ ωⱼ |βⱼ|;
# - feasible_lasso!, the lasso of Belloni, Chen, Chernozhukov and Hansen (2012)
#   for heteroscedastic noise, whose penalty loadings are estimated from the
#   residuals.
#
#     julia --project=examples examples/04_sqrt_scaled_feasible.jl

using HighDimensionalModels, ProximalBase
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(4)

n, p = 200, 400
X = randn(n, p)
β = zeros(p)
β[1:5] = [2.0, -1.5, 1.0, -1.0, 0.75]
nonzeros(b) = count(!iszero, b)

# Φ⁻¹(1 - q), the upper q-quantile of the standard normal distribution, for
# small q, from the tail approximation q ≈ φ(z) (1 - 1/z²) / z; it is within
# 0.1% for q ≤ 0.001. With Distributions.jl, use quantile(Normal(), 1 - q).
function zquantile(q)
  L = 2 * log(1 / q)
  z = sqrt(L)
  for _ in 1:50
    z = sqrt(L - 2 * log(z) - log(2π) + 2 * log(1 - 1 / z^2))
  end
  z
end

println("== Square-root lasso")

# With the loadings ωⱼ = standard deviation of X[:, j] (standardize_x = true,
# the default of sqrt_lasso), λ = 1.1 Φ⁻¹(1 - α / (2p)) sets every coefficient
# to zero with probability about 1 - α when β = 0, whatever σ. The lasso with
# λ = √(2 log(p) / n), which is right for σ = 1, does better at σ = 1, where
# its penalty is a little smaller, but selects too many variables at larger σ
# and shrinks too much at smaller σ.
λsqrt = 1.1 * zquantile(0.05 / (2p))
λlasso = sqrt(2 * log(p) / n)
println("         square-root lasso    lasso, λ for σ = 1")
println("    σ  nonzeros  ‖β̂ - β‖     nonzeros  ‖β̂ - β‖")
for σ in (0.25, 1.0, 2.0)
  yσ = X * β + σ * randn(n)
  a = sqrt_lasso(X, yσ, λsqrt)
  b = lasso(X, yσ, λlasso)
  @printf("%5.2f  %8d  %7.3f     %8d  %7.3f\n", σ, nonzeros(a.x), norm(a.x - β), nonzeros(b.x), norm(b.x - β))
end

# Like lasso, sqrt_lasso takes an intercept, unstandardized columns, explicit
# loadings and solver options
y = 3.0 .+ X * β + randn(n)
sol = sqrt_lasso(X, y, λsqrt; intercept = true)
@printf("with intercept = true: β̂₀ = %.2f, %d nonzeros\n", sol.intercept, nonzeros(sol.x))
sol = sqrt_lasso(X, y .- 3.0, λsqrt, ones(p), CDOptions(; opt_tol = 1e-10))
@printf("with ωⱼ = 1 and opt_tol = 1e-10: %d nonzeros\n", nonzeros(sol.x))

println("\n== Scaled lasso")

# scaled_lasso(X, y, λ, ω) minimizes ‖y - Xβ‖² / (2nσ) + σ/2 + λ ∑ⱼ ωⱼ |βⱼ| over
# β and σ. The minimum over σ is at σ̂ = ‖y - Xβ‖ / √n, which leaves the
# square-root lasso ‖y - Xβ‖ / √n + λ ∑ⱼ ωⱼ |βⱼ|, solved by one run of
# coordinate descent. λ is the penalty for unit noise, here √(2 log(p) / n), and
# sol.σ is the estimate of σ.
ω = ones(p)
println("    σ     σ̂  nonzeros  ‖β̂ - β‖")
for σ in (0.25, 1.0, 2.0)
  yσ = X * β + σ * randn(n)
  est = scaled_lasso(X, yσ, λlasso, ω)
  @printf("%5.2f  %5.3f  %8d  %7.3f\n", σ, est.σ, nonzeros(est.x), norm(est.x - β))
end

# Of IterLassoOptions, scaled_lasso uses only options_cd, the CDOptions of the
# coordinate descent; sol.iterations counts its passes.
y = X * β + randn(n)
println("\n  options                       σ̂         passes  converged")
for (name, options) in (("defaults", IterLassoOptions()),
                        ("opt_tol = 1e-10", IterLassoOptions(; options_cd = CDOptions(; opt_tol = 1e-10))))
  est = scaled_lasso(X, y, λlasso, ω, options)
  @printf("  %-28s  %.6f  %6d  %s\n", name, est.σ, est.iterations, est.converged)
end

# a warm start from a previous solution takes fewer passes
fit = scaled_lasso(X, y, λlasso, ω)
x = copy(fit.x)
warm = scaled_lasso!(x, X, y, λlasso, ω)
@printf("  %-28s  %.6f  %6d  %s\n", "warm start from that", warm.σ, warm.iterations, warm.converged)

# From the sufficient statistics A = X'X/n, b = -X'y/n and c = y'y/n, without
# the data: a coordinate update costs O(p) instead of O(n), which pays off when
# n is large or when A is shared, as in the precision matrix estimator of Sun
# and Zhang (2013), where it is the covariance matrix and each variable is
# regressed on the others (see the docstring of scaled_lasso!).
suff = scaled_lasso!(SparseIterate(p), CDSqrtQuadraticLoss(X'X / n, -X'y / n, dot(y, y) / n), λlasso, ω)
@printf("  %-28s  %.6f  %6d  %s   (largest difference %.1e)\n", "from A, b and c", suff.σ,
        suff.iterations, suff.converged, maximum(abs, Vector(suff.x) - Vector(fit.x)))

println("\n== Feasible lasso for heteroscedastic noise")

# The lasso is right not to select variable j when its penalty dominates the
# noise in its score, |X[:, j]'ε| / n. With heteroscedastic noise, the
# standard deviations √(∑ᵢ Xᵢⱼ² σᵢ²) / n of the scores differ between the
# variables, and one penalty level for all of them is too small for some and
# too large for others. feasible_lasso! penalizes λ0 ∑ⱼ Γⱼ |βⱼ| with loadings
# Γⱼ = √(∑ᵢ Xᵢⱼ² rᵢ² / n) estimated from the residuals r, alternating the lasso
# with the update of Γ, and λ0 = 1.1 Φ⁻¹(1 - γ / (2p)) / √n with
# γ = 0.1 / log(n). sol.penalty.w holds the loadings.
#
# Here the noise is large when |X[:, 6]| is large, although variable 6 does
# not enter the mean, so its score varies more than those of the other
# variables that do not matter, and it gets a larger loading, as with the
# oracle loadings √(∑ᵢ Xᵢⱼ² σᵢ² / n). The estimated loadings are larger than
# the oracle ones because the residuals also contain the part of Xβ that the
# lasso shrinks away, most so for variables with large coefficients, such as
# variable 1.
λ0 = 1.1 * zquantile(0.1 / log(n) / (2p)) / sqrt(n)
σi = 0.5 .+ abs.(X[:, 6])
y = X * β + σi .* randn(n)
fit = feasible_lasso!(SparseIterate(p), X, y, λ0)
oracle = vec(sqrt.(mean(abs2, X .* σi; dims = 1)))
println("selects ", findall(!iszero, fit.x), " with λ0 = ", round(λ0; digits = 3))
println("  variable j           ", join(lpad.(1:8, 6)))
println("  loading Γⱼ           ", join(lpad.(round.(fit.penalty.w[1:8]; digits = 2), 6)))
println("  oracle loading       ", join(lpad.(round.(oracle[1:8]; digits = 2), 6)))

# IterLassoOptions controls the outer iteration: the first residuals come from
# least squares on the sinit columns most correlated with y (:Screening, the
# default), from a lasso with penalty λ0 σinit ∑ⱼ sⱼ |βⱼ|, sⱼ the standard
# deviation of X[:, j] (:InitStd), or from the x passed in (:WarmStart); the
# loadings are updated until their relative change is below opt_tol, at most
# max_iter times.
println("\n  options                        nonzeros  passes  converged")
for (name, options) in (("defaults", IterLassoOptions()),
                        (":InitStd, σinit = 1", IterLassoOptions(; init_procedure = :InitStd, σinit = 1)),
                        ("opt_tol = 1e-6, max_iter = 100", IterLassoOptions(; opt_tol = 1e-6, max_iter = 100)),
                        (":WarmStart from the first fit", IterLassoOptions(; init_procedure = :WarmStart)))
  x0 = options.init_procedure == :WarmStart ? copy(fit.x) : SparseIterate(p)
  est = feasible_lasso!(x0, X, y, λ0, options)
  @printf("  %-29s  %8d  %6d  %s\n", name, nonzeros(est.x), est.iterations, est.converged)
end
