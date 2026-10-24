####################################
#
#  Precompilation
#
####################################

# The estimators' first calls, compiled into the package image; the coordinate
# descent they run is compiled in ProximalSolvers' image. PrecompileTools runs
# this workload when the package is precompiled; the preference
# `precompile_workload = false` skips it.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  n, p = 40, 12
  X = [cos(1.0 * (i + 3j)) + 0.1 * sin(1.0 * i * j) for i in 1:n, j in 1:p]
  y = X[:, 1] - X[:, 2] .+ [0.1 * sin(2.0 * i) for i in 1:n]
  z = [i / n for i in 1:n]
  @compile_workload begin
    lasso(X, y, 0.1)
    lasso(X, y, 0.1; intercept = true, standardizeX = true)
    lasso(X, y, 0.1, ones(p))
    sqrtLasso(X, y, 0.5)
    scaledLasso(X, y, 0.3, ones(p))
    feasibleLasso!(SparseIterate(p), X, y, 0.1)
    path = LassoPath(X, y; nλ = 5, intercept = true)
    LassoPath(X, y; nλ = 5, intercept = true, alpha = 0.5)
    refitLassoPath(path, X, y)
    glmLasso(X, Float64.(y .> 0), 0.05)
    coordinateDescent!(SparseIterate(p), CDLeastSquaresLoss(y, X), NormL1(0.1), CDOptions())
    coordinateDescent!(SparseIterate(p), CDQuadraticLoss(X'X / n, -X'y / n), NormL1(0.1), CDOptions())
    locpoly(X[:, 1:2], z, y, [0.5], 1, GaussianKernel(0.2))
    locpolyl1(X[:, 1:3], z, y, [0.5], 1, GaussianKernel(0.2), 0.05, true)
  end
end
