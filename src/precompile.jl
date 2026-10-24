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
  @compile_workload begin
    lasso(X, y, 0.1)
    lasso(X, y, 0.1; intercept = true, standardize_x = true)
    lasso(X, y, 0.1, ones(p))
    sqrt_lasso(X, y, 0.5)
    scaled_lasso(X, y, 0.3, ones(p))
    feasible_lasso!(SparseIterate(p), X, y, 0.1)
    path = LassoPath(X, y; nλ = 5, intercept = true)
    LassoPath(X, y; nλ = 5, intercept = true, alpha = 0.5)
    refit_lasso_path(path, X, y)
    glm_lasso(X, Float64.(y .> 0), 0.05)
    coordinate_descent!(SparseIterate(p), CDLeastSquaresLoss(y, X), NormL1(0.1), CDOptions())
    coordinate_descent!(SparseIterate(p), CDQuadraticLoss(X'X / n, -X'y / n), NormL1(0.1), CDOptions())
  end
end
