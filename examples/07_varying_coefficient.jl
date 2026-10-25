# Varying-coefficient models
#
#     yᵢ = ∑ⱼ Xᵢⱼ βⱼ(zᵢ) + εᵢ
#
# by local polynomial regression: near a point z₀, each βⱼ(z) is approximated
# by a polynomial of degree q in z - z₀, fitted by kernel-weighted least squares
# (varying_coefficient), or with a group lasso penalty that selects the
# variables when there are many and few of the βⱼ are nonzero
# (varying_coefficient_lasso). The kernels are those of KernelSmoothing.jl.
#
#     julia --project=examples examples/07_varying_coefficient.jl

using HighDimensionalModels
using KernelSmoothing: EpanechnikovKernel, GaussianKernel
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(7)

# three coefficient functions: an intercept sin(2πz), a hump 4z(1 - z) and a
# constant, with z uniform on [0, 1]
n = 500
z = rand(n)
X = [ones(n) randn(n, 2)]
βtrue = [z -> sin(2π * z), z -> 4z * (1 - z), z -> 1.0]
y = sum(X[:, j] .* βtrue[j].(z) for j in 1:3) + 0.3 * randn(n)

println("== varying_coefficient")

# varying_coefficient(X, z, y, zgrid; kernel, bandwidth, degree) fits the local
# polynomials at each point of zgrid. The kernel is a function of the scaled
# distance u = (zᵢ - z₀)/h: the EpanechnikovKernel gives weight to the
# observations with |zᵢ - z₀| < h. coef(fit) holds the estimates of βⱼ(z₀), a
# row per variable, and coef(fit; derivative = 1) those of their derivatives.
zgrid = 0.1:0.2:0.9
fit = varying_coefficient(X, z, y, zgrid; kernel = EpanechnikovKernel(), bandwidth = 0.15, degree = 1)
B = coef(fit)
println("   z0   β̂₁(z0) (true)    β̂₂(z0) (true)    β̂₃(z0) (true)")
for (k, z0) in enumerate(fit.zgrid)
  @printf("  %.1f", z0)
  for j in 1:3
    @printf("  %6.3f (%6.3f)", B[j, k], βtrue[j](z0))
  end
  println()
end
@printf("at z0 = 0.5: β̂₁' = %.2f (true %.2f)\n", coef(fit; derivative = 1)[1, 3], 2π * cos(π))

println("\n== Kernels, bandwidths and degrees")

# For the GaussianKernel the bandwidth is the standard deviation. Small
# bandwidths give noisy estimates and large ones biased estimates; a higher
# degree reduces the bias where the functions curve. The error is the root mean
# square error of β̂₁ on a fine grid.
zfine = 0.05:0.01:0.95
rmse(f) = sqrt(mean(abs2, coef(f)[1, :] - βtrue[1].(f.zgrid)))
println("  kernel                 degree  RMSE of β̂₁")
for (name, kernel, h) in (("Epanechnikov h = 0.05", EpanechnikovKernel(), 0.05),
                          ("Epanechnikov h = 0.15", EpanechnikovKernel(), 0.15),
                          ("Epanechnikov h = 0.40", EpanechnikovKernel(), 0.40),
                          ("Gaussian h = 0.06", GaussianKernel(), 0.06)),
    degree in (0, 1, 2)
  f = varying_coefficient(X, z, y, zfine; kernel, bandwidth = h, degree)
  @printf("  %-21s  %6d  %10.3f\n", name, degree, rmse(f))
end

# Leave-one-out cross-validation: each yᵢ is predicted from the local fit at zᵢ
# without observation i, which the fit with it gives exactly through its
# leverage, for each bandwidth.
hs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
cv = varying_coefficient_cv(X, z, y, hs; kernel = EpanechnikovKernel())
println("\nleave-one-out errors ", round.(cv.cvm; digits = 4), " for h = ", hs, ": h = ", cv.bandwidth)

println("\n== Standard errors")

# With se = true the fit computes robust (sandwich) standard errors from the
# residuals of the local fits at the observations. The intervals describe the variance of the
# local estimate, not its bias, so they cover where the bias is small.
fit = varying_coefficient(X, z, y, [0.2, 0.5, 0.8]; kernel = EpanechnikovKernel(), bandwidth = cv.bandwidth, se = true)
@printf("σ̂ = %.3f (true 0.3)\n", std(residuals(fit)))
lo, hi = confint(fit)
println("   z0  β̂₂(z0)  95% interval      true")
for (k, z0) in enumerate(fit.zgrid)
  @printf("  %.1f  %6.3f  [%5.3f, %5.3f]  %5.3f\n", z0, coef(fit)[2, k], lo[2, k], hi[2, k], βtrue[2](z0))
end

println("\n== Many variables: varying_coefficient_lasso")

# Of p = 50 variables, three have varying coefficients and the others none. At
# each point, varying_coefficient_lasso solves the local least squares problem
# with the group lasso penalty λ ∑ⱼ √(q + 1) ‖fitted values of variable j‖, so that a
# variable enters with all its polynomial terms, and refits the selected
# variables without the penalty. The loss is the mean squared error on the local
# sample, so λ is on the scale of an ordinary lasso.
p = 50
Xh = randn(n, p)
yh = Xh[:, 1] .* sin.(2π * z) + Xh[:, 2] .* 4z .* (1 .- z) + Xh[:, 3] + 0.3 * randn(n)
fl = varying_coefficient_lasso(Xh, z, yh, 0.1:0.2:0.9, 0.1; kernel = EpanechnikovKernel(), bandwidth = 0.2)
println("   z0  selected  β̂₁ lasso  refit  true")
for (k, z0) in enumerate(fl.zgrid)
  @printf("  %.1f  %-12s  %8.3f  %5.3f  %5.3f\n", z0, string(findall(fl.selected[:, k])), coef(fl)[1, k],
          coef(fl; refit = true)[1, k], βtrue[1](z0))
end

# The bandwidth of the lasso can be chosen by leave-one-out cross-validation as
# well, predicting yᵢ from the refit at zᵢ without observation i. It solves one
# local group lasso per observation and bandwidth, so this uses the first 200
# observations.
m = 200
cvl = varying_coefficient_lasso_cv(Xh[1:m, :], z[1:m], yh[1:m], [0.1, 0.2, 0.4], 0.1; kernel = EpanechnikovKernel())
println("\nleave-one-out errors of the lasso for h = 0.1, 0.2, 0.4: ", round.(cvl.cvm; digits = 4))
