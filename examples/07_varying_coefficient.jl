# Varying-coefficient models
#
#     yᵢ = ∑ⱼ Xᵢⱼ βⱼ(zᵢ) + εᵢ
#
# by local polynomial regression: near a point z0, each βⱼ(z) is approximated
# by a polynomial of degree q in z - z0, fit by kernel-weighted least squares
# (locpoly), or by the kernel-weighted lasso when there are many variables and
# few of the βⱼ are nonzero (locpolyl1).
#
# The estimate at z0 is a vector of p(q + 1) coefficients ordered by
# variable: entries (j - 1)(q + 1) + 1 to j(q + 1) hold βⱼ(z0), βⱼ'(z0),
# βⱼ''(z0) / 2, and so on. The functions that are not exported are called
# with the CoordinateDescent. prefix.
#
#     julia --project=examples examples/07_varying_coefficient.jl

using CoordinateDescent
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(7)

# three coefficient functions: an intercept sin(2πz), a hump 4z(1 - z) and a
# constant, with z uniform on [0, 1]
n = 500
z = rand(n)
X = [ones(n) randn(n, 2)]
βtrue = [z -> sin(2π * z), z -> 4z * (1 - z), z -> 1.0]
y = sum(X[:, j] .* βtrue[j].(z) for j in 1:3) + 0.3 * randn(n)

println("== locpoly")

# locpoly(X, z, y, zgrid, degree, kernel) returns a p(q + 1) × length(zgrid)
# matrix, and locpoly(X, z, y, z0, degree, kernel) the vector at one point.
# EpanechnikovKernel(h) has bandwidth h: it gives weight to the observations
# with |zᵢ - z0| < h.
zgrid = collect(0.1:0.2:0.9)
degree = 1
B = locpoly(X, z, y, zgrid, degree, EpanechnikovKernel(0.15))
levels = 1:(degree + 1):size(B, 1)                  # the rows of β₁(z0), β₂(z0), β₃(z0)
println("   z0   β̂₁(z0) (true)    β̂₂(z0) (true)    β̂₃(z0) (true)")
for (k, z0) in enumerate(zgrid)
  @printf("  %.1f", z0)
  for j in 1:3
    @printf("  %6.3f (%6.3f)", B[levels[j], k], βtrue[j](z0))
  end
  println()
end
b = locpoly(X, z, y, 0.5, degree, EpanechnikovKernel(0.15))
@printf("at z0 = 0.5 alone: β̂₁ = %.3f, and its derivative β̂₁' = %.2f (true %.2f)\n", b[1], b[2], 2π * cos(π))

println("\n== Kernels, bandwidths and degrees")

# GaussianKernel(h) is exp(-(zᵢ - z0)² / h) / h: h is a variance, 2σ² for a
# kernel with standard deviation σ. Small bandwidths give noisy estimates and
# large ones biased estimates; a higher degree reduces the bias where the
# functions curve. The error is the root mean square error of β̂₁ on a fine
# grid.
zfine = collect(0.05:0.01:0.95)
rmse(B) = sqrt(mean(abs2, B[1, :] - βtrue[1].(zfine)))
println("  kernel                 degree  RMSE of β̂₁")
for (name, kernel) in (("Epanechnikov h = 0.05", EpanechnikovKernel(0.05)),
                       ("Epanechnikov h = 0.15", EpanechnikovKernel(0.15)),
                       ("Epanechnikov h = 0.40", EpanechnikovKernel(0.40)),
                       ("Gaussian σ = 0.06", GaussianKernel(2 * 0.06^2))),
    degree in (0, 1, 2)
  @printf("  %-21s  %6d  %10.3f\n", name, degree, rmse(locpoly(X, z, y, zfine, degree, kernel)))
end

# Leave-one-out cross-validation: lvocv_locpoly predicts each yᵢ from the fit
# at zᵢ without observation i, for each bandwidth in hs, and returns the sums
# of squared errors. createKernel(KernelType, h) is how it builds the kernels.
hs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
cv = CoordinateDescent.lvocv_locpoly(X, z, y, 1, hs, EpanechnikovKernel{Float64})
h = hs[argmin(cv)]
println("\nleave-one-out errors ", round.(cv / n; digits = 4), " for h = ", hs, ": h = $h")

println("\n== Standard errors")

# getStandardError(X, z, σ², z0, degree, kernel) returns the variances of
# β̂ⱼ(z0) under homoscedastic noise with variance σ², and getStandardErrorHEW
# with the squared residuals in place of σ² is robust to heteroscedasticity.
# getResiduals! computes the residuals of a fit on a grid, interpolating the
# coefficients linearly between the grid points.
kernel = EpanechnikovKernel(h)
Bfine = locpoly(X, z, y, zfine, 1, kernel)
r = CoordinateDescent.getResiduals!(zeros(n), X, z, y, zfine, Bfine, 1)
σ2 = mean(abs2, r)
@printf("σ̂ = %.3f (true 0.3)\n", sqrt(σ2))
println("   z0  β̂₂(z0)  95% interval      true  robust interval")
for z0 in (0.2, 0.5, 0.8)
  bz = locpoly(X, z, y, z0, 1, kernel)
  se = sqrt.(CoordinateDescent.getStandardError(X, z, σ2, z0, 1, kernel))
  seHEW = sqrt.(CoordinateDescent.getStandardErrorHEW(X, z, r .^ 2, z0, 1, kernel))
  @printf("  %.1f  %6.3f  [%5.3f, %5.3f]  %5.3f  [%5.3f, %5.3f]\n", z0, bz[3], bz[3] - 1.96se[2], bz[3] + 1.96se[2],
          βtrue[2](z0), bz[3] - 1.96seHEW[2], bz[3] + 1.96seHEW[2])
end

println("\n== Many variables: locpolyl1")

# Of p = 50 variables, three have varying coefficients and the others none.
# locpolyl1(X, z, y, zgrid, degree, kernel, λ0, refit) solves at each grid
# point the weighted lasso
#     ∑ᵢ wᵢ (yᵢ - X̃ᵢ'β)² / (2n) + λ0 ∑ₖ sₖ |βₖ|,
# with wᵢ = K_h(zᵢ, z0), the expanded design X̃ and sₖ the weighted standard
# deviations of its columns, each problem warm started from the previous grid
# point. It returns sparse matrices B and BR, with BR the weighted least
# squares refit on the selected variables when refit is true. Options of
# coordinate descent go last.
p = 50
Xh = randn(n, p)
yh = Xh[:, 1] .* sin.(2π * z) + Xh[:, 2] .* 4z .* (1 .- z) + Xh[:, 3] + 0.3 * randn(n)
zgrid = collect(0.1:0.2:0.9)
B, BR = locpolyl1(Xh, z, yh, zgrid, 1, EpanechnikovKernel(0.2), 0.1, true, CDOptions(; optTol = 1e-8))
println("   z0  selected  β̂₁ lasso  refit  true")
for (k, z0) in enumerate(zgrid)
  S = CoordinateDescent.get_nonzero_coordinates(B[:, k], p, 1, false)
  @printf("  %.1f  %-8s  %8.3f  %5.3f  %5.3f\n", z0, string(findall(S)), B[1, k], BR[1, k], βtrue[1](z0))
end

# The bandwidth of locpolyl1 can be chosen by leave-one-out cross-validation
# as well: lvocv_locpolyl1 fits at each zᵢ without observation i, with the
# penalty λ0 σ̂ and the noise level σ̂ estimated as in the scaled lasso, and
# predicts yᵢ from the least squares refit on the selected variables. Here
# σ̂ ≈ 0.3, so λ0 = 0.3 gives about the penalty used above. A λ0 that is too
# small selects more variables than the observations near zᵢ can refit, and
# the errors blow up. It solves n lasso problems per bandwidth, so this uses
# the first 200 observations.
m = 200
cvl = CoordinateDescent.lvocv_locpolyl1(Xh[1:m, :], z[1:m], yh[1:m], 1, [0.1, 0.2, 0.4],
                                        EpanechnikovKernel{Float64}, 0.3)
println("\nleave-one-out errors of locpolyl1 for h = 0.1, 0.2, 0.4: ", round.(cvl / m; digits = 4))
