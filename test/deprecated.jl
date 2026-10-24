# Run with --depwarn=yes even when the canonical suite uses --depwarn=error.
using CoordinateDescent, ProximalBase, ProximalSolvers, Test, Random, LinearAlgebra

@testset "deprecated names execute their forwards" begin
  X = randn(Xoshiro(2), 40, 4)
  y = X * [1., -1., 0., 0.] + 0.1 * randn(Xoshiro(3), 40)
  @test_deprecated r"max_iter" CDOptions(; maxIter=100)
  o = @test_deprecated r"opt_tol" CDOptions(CDOptions(); optTol=1e-9)
  @test o.opt_tol == 1e-9
  @test_deprecated r"warm_start" CDOptions(; warmStart=false)
  @test_deprecated r"num_steps" CDOptions(; numSteps=10)
  @test_deprecated r"relative_tol" CDOptions(; relativeTol=false)
  io = @test_deprecated r"init_procedure" IterLassoOptions(; initProcedure=:InitStd)
  @test io.init_procedure === :InitStd
  @test_deprecated r"options_cd" IterLassoOptions(IterLassoOptions(); optionsCD=o)
  @test_deprecated r"max_iter" o.maxIter
  @test_deprecated r"opt_tol" io.optTol
  @test_throws ArgumentError CDOptions(; maxIter=10, max_iter=20)
  @test_deprecated r"standardize_x" lasso(X, y, 0.1; standardizeX=false)
  @test_throws ArgumentError lasso(X, y, 0.1; standardizeX=false, standardize_x=false)

  for (old, new, args) in (
      (sqrtLasso, sqrt_lasso, (X, y, 0.1)),
      (scaledLasso, scaled_lasso, (X, y, 0.1, ones(4))),
      (adaptiveLasso, adaptive_lasso, (X, y, 0.1)),
      (groupLasso, group_lasso, (X, y, 0.1, [1:2, 3:4])),
      (sparseGroupLasso, sparse_group_lasso, (X, y, 0.1, [1:2, 3:4])),
      (multiLasso, multi_lasso, (X, hcat(y, y), 0.1)),
      (postLasso, post_lasso, (X, y, lasso(X, y, 0.1))))
    Random.seed!(3)
    legacy = @test_deprecated old(args...)
    Random.seed!(3)
    current = new(args...)
    @test coef(legacy) ≈ coef(current)
  end
  @test_deprecated r"theory_lambda" theoryLambda(40, 4)
  @test_deprecated r"scaled_lasso!" scaledLasso!(SparseIterate(4), X, y, 0.1, ones(4))
  @test_deprecated r"feasible_lasso!" feasibleLasso!(SparseIterate(4), X, y, 0.2)
  @test_deprecated r"glm_lasso" glmLasso(X, Float64.(y .> 0), 0.1)
  @test_deprecated r"newton_tol" glm_lasso(X, Float64.(y .> 0), 0.1; newtonTol=1e-8)
  @test_deprecated r"newton_max_iter" GLMLassoPath(X, Float64.(y .> 0), [0.2, 0.1]; newtonMaxIter=100)
  @test_deprecated r"max_iter" rlasso(X, y; maxIter=15)
  @test_deprecated r"double_selection" doubleSelection(y, randn(Xoshiro(7), 40), X)
  z = collect(range(0., 1.; length=40))
  kernel = u -> exp(-u*u/2) / sqrt(2pi)
  @test_deprecated r"varying_coefficient" varyingCoefficient(X, z, y, [0.3, 0.7]; kernel, bandwidth=0.3)
  @test_deprecated r"varying_coefficient_cv" varyingCoefficientCV(X, z, y, [0.2, 0.3]; kernel)
  @test_deprecated r"varying_coefficient_lasso" varyingCoefficientLasso(X, z, y, [0.3, 0.7], 0.1; kernel, bandwidth=0.3)
  @test_deprecated r"varying_coefficient_lasso_cv" varyingCoefficientLassoCV(X, z, y, [0.3], 0.1; kernel)
  x = SparseIterate(4)
  f, g = CDLeastSquaresLoss(y, X), NormL1(0.1)
  result = @test_deprecated r"coordinateDescent!" coordinateDescent!(x, f, g)
  @test result === x
  info = CDInfo()
  @test_deprecated r"coordinateDescent!" coordinateDescent!(x, f, g, CDOptions(), info)
  @test info.converged
  @test_deprecated r"max_change" info.maxChange
  @test_deprecated r"max_change" (info.maxChange = 0.5)
  @test info.max_change == 0.5
  path = LassoPath(X, y, [0.2, 0.1])
  @test_deprecated r"kkt_checks" path.kktChecks
  old = @test_deprecated refitLassoPath(path, X, y)
  @test old == refit_lasso_path(path, X, y)
end

# Deprecated interface bindings must remain the same generic functions so
# downstream packages can still extend the old spelling.
struct LegacyLoss <: CoordinateDifferentiableFunction
  c::Vector{Float64}
end
CoordinateDescent.initialize!(::LegacyLoss, x) = nothing
CoordinateDescent.gradient(f::LegacyLoss, x, k) = x[k] - f.c[k]
CoordinateDescent.numCoordinates(f::LegacyLoss) = length(f.c)
function CoordinateDescent.descendCoordinate!(f::LegacyLoss, g, x, k)
  old = x[k]
  x[k] = f.c[k]
  prox_coordinate!(x, g, k, 1.) - old
end
CoordinateDescent.coordinateScale(::LegacyLoss, k) = 1.
CoordinateDescent.responseScale(::LegacyLoss) = 1.
function CoordinateDescent.fullGradient!(grad, f::LegacyLoss, x)
  grad .= Vector(x) .- f.c
end
@testset "deprecated interface bindings" begin
  for (old, new) in ((:numCoordinates, :num_coordinates), (:descendCoordinate!, :descend_coordinate!),
                     (:coordinateScale, :coordinate_scale), (:responseScale, :response_scale),
                     (:fullGradient!, :full_gradient!))
    @test Base.isdeprecated(CoordinateDescent, old)
    @test getfield(CoordinateDescent, old) === getfield(CoordinateDescent, new)
  end
  x = SparseIterate(3)
  r = coordinate_descent!(x, LegacyLoss([2., -3., 0.1]), NormL1(1.), CDOptions(; warm_start=false))
  @test r.converged
  @test Vector(x) ≈ [1., -2., 0.]
end
