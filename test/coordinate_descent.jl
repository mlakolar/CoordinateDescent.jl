module CoordinateDescentTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays
using Statistics

Random.seed!(1)

# test |Y - β|_2^2 + λ⋅|β|_1
@testset "Small ProxL1" begin
  Y = [1., 1.5]

  λ = 1.2
  g = ProximalBase.ProxL1(λ)
  f = CDQuadraticLoss(Matrix(1.0I, 2, 2), -Y)

  opt = CDOptions(;maxIter=100, optTol=1e-8, warmStart=true, randomize=false)
  x = SparseIterate(2)
  coordinateDescent!(x, f, g, opt)

  @test Vector(x) ≈ Vector([0., 0.3])
end


# check that the warm start and non warm start produce the same result
@testset "ProxL1" begin

    for i=1:NUMBER_REPEAT
      n = 500
      p = 50
      s = 5

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 0.02
      g = ProximalBase.ProxL1(λ)
      f = CDLeastSquaresLoss(Y, X)

      opt1 = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=true, randomize=false)
      opt2 = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=true, randomize=true)
      opt3 = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=false, randomize=false)
      opt4 = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=false, randomize=true)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      coordinateDescent!(x1, f, g, opt1)
      coordinateDescent!(x2, f, g, opt2)
      coordinateDescent!(x3, f, g, opt3)
      coordinateDescent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-5
      @test Vector(x3) ≈ Vector(x2) atol=1e-5
      @test Vector(x4) ≈ Vector(x2) atol=1e-5
    end
end

@testset "AProxL1" begin

    for i=1:NUMBER_REPEAT
      n = 500
      p = 50
      s = 10

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 0.01
      g = ProximalBase.ProxL1(λ, rand(p))
      f = CDLeastSquaresLoss(Y, X)

      opt1 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=false)
      opt2 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=true)
      opt3 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=false)
      opt4 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=true)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      coordinateDescent!(x1, f, g, opt1)
      coordinateDescent!(x2, f, g, opt2)
      coordinateDescent!(x3, f, g, opt3)
      coordinateDescent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-5
      @test Vector(x3) ≈ Vector(x2) atol=1e-5
      @test Vector(x4) ≈ Vector(x2) atol=1e-5
    end
end

# path following must end at the target λ, not one step short of it
@testset "path following reaches λ" begin
  n, p, s = 500, 50, 5

  for seed in 1:5
    Random.seed!(seed)
    X = randn(n, p)
    Y = X[:,1:s] * randn(s) + randn(n)

    λ = 0.02
    for g in (ProximalBase.ProxL1(λ), ProximalBase.ProxL1(1., fill(λ, p)))
      f = CDLeastSquaresLoss(Y, X)
      x = SparseIterate(p)
      coordinateDescent!(x, f, g, CDOptions(;maxIter=5000, optTol=1e-12, warmStart=false))
      @test maximum(abs.(X'*(Y - X*x) / n)) ≈ λ atol=1e-8
    end
  end
end

@testset "ProxZero" begin
  n, p = 200, 10
  X = randn(n, p)
  Y = X * randn(p) + randn(n)

  for warmStart in (true, false)
    opt = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=warmStart)

    x = SparseIterate(p)
    coordinateDescent!(x, CDLeastSquaresLoss(Y, X), ProxZero(), opt)
    @test Vector(x) ≈ X \ Y atol=1e-6

    x = SparseIterate(p)
    coordinateDescent!(x, CDQuadraticLoss(X'X/n, -X'Y/n), ProxZero(), opt)
    @test Vector(x) ≈ X \ Y atol=1e-6
  end
end

@testset "weighted least squares loss" begin
  n, p, s = 500, 50, 5
  X = randn(n, p)
  Y = X[:,1:s] * randn(s) + randn(n)
  w = 2 * rand(n)
  sw = sqrt.(w)

  g = ProximalBase.ProxL1(0.05)
  f1 = CDLeastSquaresLoss(sw .* Y, sw .* X)
  f2 = CDWeightedLSLoss(Y, X, w)

  for warmStart in (true, false), randomize in (true, false)
    opt = CDOptions(;maxIter=5000, optTol=1e-12, warmStart=warmStart, randomize=randomize)
    x1 = SparseIterate(sprand(p, 0.6))
    x2 = SparseIterate(sprand(p, 0.6))
    coordinateDescent!(x1, f1, g, opt)
    coordinateDescent!(x2, f2, g, opt)
    @test Vector(x1) ≈ Vector(x2) atol=1e-6
  end
end

@testset "Float32" begin
  n, p, s = 200, 20, 5
  X = randn(Float32, n, p)
  Y = X[:,1:s] * ones(Float32, s) + 0.1f0 * randn(Float32, n)

  g = ProximalBase.ProxL1(0.1f0)
  for f in (CDLeastSquaresLoss(Y, X), CDQuadraticLoss(X'X/n, -X'Y/n))
    x = SparseIterate(Float32, p)
    coordinateDescent!(x, f, g, CDOptions(;optTol=1e-6))
    @test eltype(x) == Float32
    @test maximum(abs.(X'*(Y - X*x) / n)) ≈ 0.1f0 atol=1e-4
  end
end

@testset "argument checks" begin
  n, p = 20, 5
  X = randn(n, p)
  Y = randn(n)

  @test_throws DimensionMismatch CDLeastSquaresLoss(randn(n+1), X)
  @test_throws DimensionMismatch CDWeightedLSLoss(Y, X, rand(n+1))
  @test_throws DimensionMismatch CDSqrtLassoLoss(randn(n+1), X)
  @test_throws ArgumentError CDQuadraticLoss(randn(p, p) + triu(ones(p, p)), randn(p))
  @test_throws ArgumentError CDQuadraticLoss(Matrix(1.0I, p, p), randn(p+1))

  f = CDLeastSquaresLoss(Y, X)
  @test_throws DimensionMismatch coordinateDescent!(SparseIterate(p+1), f, ProximalBase.ProxL1(0.1))
  @test_throws DimensionMismatch coordinateDescent!(SparseIterate(p+1), f, ProxZero())
  @test_throws DimensionMismatch coordinateDescent!(SparseIterate(p), f, ProximalBase.ProxL1(0.1, ones(p+1)))
end

@testset "gradients" begin
  n, p = 50, 8
  X = randn(n, p)
  Y = randn(n)
  w = rand(n)
  x = SparseIterate(sprand(p, 0.5))
  r = Y - X * Vector(x)

  f = CDLeastSquaresLoss(Y, X)
  CoordinateDescent.initialize!(f, x)
  @test CoordinateDescent.numCoordinates(f) == p
  @test [CoordinateDescent.gradient(f, x, k) for k=1:p] ≈ -X'r / n

  f = CDWeightedLSLoss(Y, X, w)
  CoordinateDescent.initialize!(f, x)
  @test CoordinateDescent.numCoordinates(f) == p
  @test [CoordinateDescent.gradient(f, x, k) for k=1:p] ≈ -X'*(w .* r) / n

  f = CDSqrtLassoLoss(Y, X)
  CoordinateDescent.initialize!(f, x)
  @test CoordinateDescent.numCoordinates(f) == p
  @test [CoordinateDescent.gradient(f, x, k) for k=1:p] ≈ -X'r / norm(r)

  A = X'X / n
  b = randn(p)
  f = CDQuadraticLoss(A, b)
  CoordinateDescent.initialize!(f, x)
  @test CoordinateDescent.numCoordinates(f) == p
  @test [CoordinateDescent.gradient(f, x, k) for k=1:p] ≈ A * Vector(x) + b
end

# a user-defined loss f(x) = ‖x - c‖² / 2 through the documented interface;
# with ProxL1 its minimizer is the soft-thresholded c
struct SeparableLoss <: CoordinateDifferentiableFunction
  c::Vector{Float64}
end
CoordinateDescent.initialize!(::SeparableLoss, x) = nothing
CoordinateDescent.numCoordinates(f::SeparableLoss) = length(f.c)
CoordinateDescent.gradient(f::SeparableLoss, x, k) = x[k] - f.c[k]
function CoordinateDescent.descendCoordinate!(f::SeparableLoss, g::ProxL1, x, k)
  oldVal = x[k]
  x[k] = f.c[k]
  cdprox!(g, x, k, 1.)
  x[k] - oldVal
end

@testset "user-defined loss" begin
  c = [3., -0.5, 0.2, -2.]
  for warmStart in (true, false)
    x = SparseIterate(4)
    coordinateDescent!(x, SeparableLoss(c), ProximalBase.ProxL1(1.), CDOptions(;warmStart=warmStart))
    @test Vector(x) ≈ [2., 0., 0., -1.]
    @test nnz(x) == 2
  end

  # other iterates are not screened along the path of warmStart = false
  c = [3., -0.5, 0.2, -2., 1.5, 0.7]
  for warmStart in (true, false)
    x = SymmetricSparseIterate(3)
    coordinateDescent!(x, SeparableLoss(c), ProximalBase.ProxL1(1.), CDOptions(;warmStart=warmStart))
    @test [x[k] for k in 1:6] ≈ [2., 0., 0., -1., 0.5, 0.]
  end

  # with penalty weights, the path starts from max |cₖ| / wₖ over the penalized
  # coordinates, 2 / 0.5 here
  w = [1., 0., 2., 0.5, 1., 1.]
  x = SymmetricSparseIterate(3)
  @test CoordinateDescent._findLambdaMax(x, SeparableLoss(c), ProximalBase.ProxL1(1., w)) == 4.
  for warmStart in (true, false)
    x = SymmetricSparseIterate(3)
    coordinateDescent!(x, SeparableLoss(c), ProximalBase.ProxL1(1., w), CDOptions(;warmStart=warmStart))
    @test [x[k] for k in 1:6] ≈ [2., -0.5, 0., -1.5, 0.5, 0.]
  end

  struct IncompleteLoss <: CoordinateDifferentiableFunction end
  @test_throws ErrorException CoordinateDescent.initialize!(IncompleteLoss(), SparseIterate(2))
  @test_throws ErrorException CoordinateDescent.gradient(IncompleteLoss(), SparseIterate(2), 1)
  @test_throws ErrorException CoordinateDescent.numCoordinates(IncompleteLoss())
  @test_throws ErrorException CoordinateDescent.descendCoordinate!(IncompleteLoss(), ProxZero(), SparseIterate(2), 1)
end

# a user-defined penalty, the elastic net λ (α ‖x‖₁ + (1 - α) ‖x‖² / 2), through
# its proximal operator on one coordinate
struct ElasticNetPenalty <: ProximableFunction
  λ::Float64
  α::Float64
end
ProximalBase.cdprox!(g::ElasticNetPenalty, x::SparseIterate{Float64}, k::Int, γ::Float64) =
  x[k] = shrink(x[k], γ * g.λ * g.α) / (1 + γ * g.λ * (1 - g.α))

@testset "user-defined penalty" begin
  n, p, s = 100, 20, 5
  X = randn(n, p)
  Y = X[:, 1:s] * ones(s) + randn(n)
  λ, α = 0.1, 0.5
  g = ElasticNetPenalty(λ, α)

  # the elastic net is the lasso on the data augmented with √(nλ(1 - α)) I,
  # whose loss divides by n + p instead of n
  Xa = [X; Matrix(sqrt(n * λ * (1 - α)) * I, p, p)]
  Ya = [Y; zeros(p)]
  xa = SparseIterate(p)
  coordinateDescent!(xa, CDLeastSquaresLoss(Ya, Xa), ProxL1(n / (n + p) * λ * α), CDOptions(;optTol=1e-12))
  @test 0 < nnz(xa) < p

  for f in (CDLeastSquaresLoss(Y, X), CDWeightedLSLoss(Y, X, ones(n)), CDQuadraticLoss(X'X / n, -X'Y / n)),
      warmStart in (true, false), randomize in (true, false)
    x = SparseIterate(sprand(p, 0.5))
    info = CDInfo()
    coordinateDescent!(x, f, g, CDOptions(;optTol=1e-12, warmStart=warmStart, randomize=randomize), info)
    @test info.converged
    @test Vector(x) ≈ Vector(xa) atol=1e-8
    @test nnz(x) == nnz(xa)                  # no stored zeros
  end

  x = zeros(p)
  coordinateDescent!(x, CDLeastSquaresLoss(Y, X), g, CDOptions(;optTol=1e-12))
  @test x ≈ Vector(xa) atol=1e-8

  # the square-root lasso loss takes the ℓ1 penalty only, and the group lasso
  # penalty needs an AtomIterate
  @test_throws ErrorException coordinateDescent!(SparseIterate(p), CDSqrtLassoLoss(Y, X), g)
  @test_throws ArgumentError coordinateDescent!(SparseIterate(p), CDLeastSquaresLoss(Y, X), ProxGroupL2(0.1, [1:10, 11:20]))
end

@testset "maxIter" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  f = CDLeastSquaresLoss(Y, X)
  g = ProximalBase.ProxL1(0.01)

  # a single pass is one sweep of coordinate updates
  x1 = SparseIterate(p)
  info = CDInfo()
  @test_logs (:warn, r"did not converge in maxIter = 1 passes") coordinateDescent!(x1, f, g, CDOptions(;maxIter=1, randomize=false), info)
  x2 = SparseIterate(p)
  CoordinateDescent.initialize!(f, x2)
  for k=1:p
    CoordinateDescent.descendCoordinate!(f, g, x2, k)
  end
  @test Vector(x1) == Vector(x2)
  @test info.iterations == 1
  @test !info.converged
end

@testset "convergence info" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  g = ProximalBase.ProxL1(0.05)

  for f in (CDLeastSquaresLoss(Y, X), CDQuadraticLoss(X'X/n, -X'Y/n)), warmStart in (true, false)
    opt = CDOptions(;optTol=1e-10, warmStart=warmStart)

    info = CDInfo()
    x = SparseIterate(p)
    @test_logs coordinateDescent!(x, f, g, opt, info)
    @test info.converged
    @test info.iterations >= 2          # a pass that moves x and one that confirms it
    @test info.maxChange < info.tolerance
    @test info.tolerance ≈ 1e-10 * CoordinateDescent.responseScale(f)
    @test maximum(abs.(X'*(Y - X*x) / n)) ≈ 0.05 atol=1e-8

    # stopped by maxIter: reported and warned about; with warmStart = false the
    # passes of the numSteps path steps are counted too (the first step, at
    # λmax, stops after one pass)
    opt = CDOptions(;maxIter=2, optTol=1e-10, warmStart=warmStart)
    x = SparseIterate(p)
    @test_logs (:warn, r"did not converge") coordinateDescent!(x, f, g, opt, info)
    @test !info.converged
    if warmStart
      @test info.iterations == 2
    else
      @test opt.numSteps < info.iterations <= 2 * (opt.numSteps + 1)
    end
    @test info.maxChange > info.tolerance
  end

  # ProxZero, and an info object can be reused
  info = CDInfo()
  x = SparseIterate(p)
  @test_logs coordinateDescent!(x, CDLeastSquaresLoss(Y, X), ProxZero(), CDOptions(;optTol=1e-10), info)
  @test info.converged
  @test Vector(x) ≈ X \ Y atol=1e-6
end

@testset "default options" begin
  opt = CDOptions()
  @test opt.maxIter == 100_000
  @test opt.optTol == 1e-7
  @test opt.randomize
  @test opt.warmStart
  @test opt.relativeTol
end

@testset "scale-free stopping rule" begin
  # the lasso with y and λ scaled by c: with powers of two the iterates scale
  # exactly, so the relative rule stops after the same number of passes
  n, p = 100, 50
  X = randn(n, p)
  Y = X[:,1:5] * ones(5) + randn(n)
  λ = 0.05
  opt = CDOptions(;randomize=false)
  ref = lasso(X, Y, λ, opt)
  for c in (2.0^-10, 2.0^10)
    sol = lasso(X, c .* Y, c * λ, opt)
    @test sol.iterations == ref.iterations
    @test Vector(sol.x) ≈ c .* Vector(ref.x)
  end
  # the absolute rule depends on c
  opt = CDOptions(;randomize=false, relativeTol=false)
  @test lasso(X, 2.0^-10 .* Y, 2.0^-10 * λ, opt).iterations < lasso(X, 2.0^10 .* Y, 2.0^10 * λ, opt).iterations

  # the scales of the built-in losses
  f = CDLeastSquaresLoss(Y, X)
  @test CoordinateDescent.responseScale(f) ≈ norm(Y) / sqrt(n)
  @test CoordinateDescent.coordinateScale(f, 3) ≈ norm(X[:, 3]) / sqrt(n)
  w = rand(n)
  f = CDWeightedLSLoss(Y, X, w)
  @test CoordinateDescent.responseScale(f) ≈ sqrt(sum(w .* Y.^2) / n)
  @test CoordinateDescent.coordinateScale(f, 3) ≈ sqrt(sum(w .* X[:, 3].^2) / n)
  f = CDSqrtLassoLoss(Y, X)
  @test CoordinateDescent.responseScale(f) ≈ norm(Y) / sqrt(n)
  @test CoordinateDescent.coordinateScale(f, 3) ≈ norm(X[:, 3]) / sqrt(n)
  f = CDQuadraticLoss(X'X / n, -X'Y / n)
  @test CoordinateDescent.coordinateScale(f, 3) ≈ norm(X[:, 3]) / sqrt(n)
  @test 0 < CoordinateDescent.responseScale(f) <= norm(Y) / sqrt(n)

  # a zero response falls back to the absolute rule, and zero is found at once
  info = CDInfo()
  x = SparseIterate(p)
  coordinateDescent!(x, CDLeastSquaresLoss(zeros(n), X), ProximalBase.ProxL1(0.1), CDOptions(), info)
  @test info.converged
  @test info.iterations == 1
  @test info.tolerance == CDOptions().optTol
  @test nnz(x) == 0
end

@testset "option conversions" begin
  opt = CDOptions(;maxIter=1e5, optTol=1, numSteps=10.0)
  @test opt.maxIter === 100_000
  @test opt.optTol === 1.0
  @test opt.numSteps === 10
  @test_throws InexactError CDOptions(;maxIter=1.5)

  opt = IterLassoOptions(;maxIter=10.0, optTol=1, sinit=3.0, σinit=2)
  @test opt.maxIter === 10
  @test opt.optTol === 1.0
  @test opt.sinit === 3
  @test opt.σinit === 2.0
end

@testset "zero penalty weights" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X[:,1:3] * ones(3) + 0.1 * randn(n)
  f = CDLeastSquaresLoss(Y, X)

  # the path of warmStart = false starts from the largest λ0 at which the
  # penalized coordinates are zero; an unpenalized coordinate made it Inf
  w = ones(p)
  w[1] = 0.
  x = SparseIterate(p)
  CoordinateDescent.initialize!(f, x)
  @test CoordinateDescent._findLambdaMax(x, f, ProximalBase.ProxL1(1., w)) ≈ maximum(abs.(X[:,2:end]'Y / n))

  for warmStart in (true, false)
    x = SparseIterate(p)
    info = CDInfo()
    coordinateDescent!(x, f, ProximalBase.ProxL1(0.1, w), CDOptions(;optTol=1e-12, warmStart=warmStart), info)
    @test info.converged
    grad = X'*(Y - X*x) / n
    @test abs(grad[1]) < 1e-8                     # unpenalized
    @test maximum(abs.(grad[2:end])) ≈ 0.1 atol=1e-8
  end

  # no penalty at all: the path is skipped and the least squares fit is returned
  for g in (ProximalBase.ProxL1(1., zeros(p)), ProximalBase.ProxL1(0.))
    x = SparseIterate(p)
    coordinateDescent!(x, f, g, CDOptions(;optTol=1e-12, warmStart=false))
    @test Vector(x) ≈ X \ Y atol=1e-6
  end
end

@testset "full gradient" begin
  n, p = 50, 20
  X = randn(n, p) .+ 1
  Y = X * randn(p) + randn(n)
  w = rand(n)
  x = SparseIterate(sprandn(p, 0.3))
  for f in (CDLeastSquaresLoss(Y, X), CDLeastSquaresLoss(Y, X; intercept=true), CDWeightedLSLoss(Y, X, w),
            CDSqrtLassoLoss(Y, X), CDSqrtLassoLoss(Y, X; intercept=true), CDQuadraticLoss(X'X / n, -X'Y / n))
    CoordinateDescent.initialize!(f, x)
    grad = CoordinateDescent.fullGradient!(zeros(p), f, x)
    @test grad ≈ [CoordinateDescent.gradient(f, x, k) for k in 1:p]
  end
  f = CDLeastSquaresLoss(Y, X; intercept=true)
  CoordinateDescent.initialize!(f, x)
  Xc = X .- mean(X, dims=1)
  @test CoordinateDescent.fullGradient!(zeros(p), f, x) ≈ -Xc' * (Y .- mean(Y) - Xc * Vector(x)) / n
end

@testset "screening corrects the strong rule" begin
  # with a strong set that is too small, the violators are found and added
  n, p = 100, 500
  X = randn(n, p)
  Y = X[:,1:20] * ones(20) + randn(n)
  f = CDLeastSquaresLoss(Y, X)
  x = SparseIterate(p)
  CoordinateDescent.initialize!(f, x)
  grad = CoordinateDescent.fullGradient!(zeros(p), f, x)
  g = ProximalBase.ProxL1(0.2 * maximum(abs, grad))
  info = CDInfo()
  opt = CDOptions(;optTol=1e-12)

  # λprev = 0 keeps only the coordinates with |∇ₖf| ≥ 2λ at zero
  checks = CoordinateDescent._screenedSolve!(x, f, g, 0.0, grad, CoordinateDescent.OrderedIterator(x), opt, info, falses(p))
  @test checks > 1
  @test info.converged
  x2 = SparseIterate(p)
  coordinateDescent!(x2, CDLeastSquaresLoss(Y, X), g, opt)
  @test Vector(x) ≈ Vector(x2) atol=1e-8
  @test grad ≈ -X'*(Y - X*x) / n                  # the gradient at the solution
end

@testset "dense starting point" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  f = CDLeastSquaresLoss(Y, X)
  g = ProximalBase.ProxL1(0.05)
  opt = CDOptions(;optTol=1e-12)

  x = SparseIterate(p)
  coordinateDescent!(x, f, g, opt)
  for x0 in (zeros(p), randn(p))
    xd = copy(x0)
    @test coordinateDescent!(xd, f, g, opt) === xd
    @test xd ≈ Vector(x) atol=1e-8
  end
end

@testset "caches match their definitions" begin
  # the losses skip the updates of coordinates that do not move, and the
  # least squares losses compute the column norms on the first visit
  n, p = 100, 30
  X = randn(n, p)
  Y = X[:,1:5] * ones(5) + randn(n)
  w = 2 * rand(n)

  for (f, g) in ((CDLeastSquaresLoss(Y, X), ProximalBase.ProxL1(0.1)),
                 (CDWeightedLSLoss(Y, X, w), ProximalBase.ProxL1(0.1)),
                 (CDSqrtLassoLoss(Y, X), ProximalBase.ProxL1(2.5)))
    x = SparseIterate(p)
    coordinateDescent!(x, f, g, CDOptions(;optTol=1e-10))
    @test 0 < nnz(x) < p
    @test nnz(x) == count(!iszero, Vector(x))     # no stored zeros
    @test f.r ≈ Y - X*x
  end
  f = CDLeastSquaresLoss(Y, X)
  coordinateDescent!(SparseIterate(p), f, ProximalBase.ProxL1(0.1))
  @test f.a ≈ vec(sum(abs2, X, dims=1))
  f = CDWeightedLSLoss(Y, X, w)
  coordinateDescent!(SparseIterate(p), f, ProximalBase.ProxL1(0.1))
  @test f.a ≈ vec(sum(abs2, X .* sqrt.(w), dims=1))

  A = X'X / n
  f = CDQuadraticLoss(A, -X'Y / n)
  x = SparseIterate(p)
  coordinateDescent!(x, f, ProximalBase.ProxL1(0.1), CDOptions(;optTol=1e-10))
  @test f.Ax ≈ A * Vector(x)
end

@testset "data changed in place" begin
  # locpolyl1 changes the weights and the design in place between solves; the
  # cached column norms must follow
  n, p = 100, 20
  X = randn(n, p)
  Y = X[:,1:3] * ones(3) + randn(n)
  w = rand(n)
  g = ProximalBase.ProxL1(0.05)
  opt = CDOptions(;optTol=1e-12)

  f = CDWeightedLSLoss(Y, X, w)
  x = SparseIterate(p)
  coordinateDescent!(x, f, g, opt)
  w .= 2 .* rand(n)
  X .*= 1.5
  coordinateDescent!(x, f, g, opt)

  x2 = SparseIterate(p)
  coordinateDescent!(x2, CDWeightedLSLoss(Y, copy(X), copy(w)), g, opt)
  @test Vector(x) ≈ Vector(x2) atol=1e-8
end


@testset "group lasso" begin
  # groups of sizes 1 to 5, four of them in the model; f(x) = ‖y - Xx‖² / (2n)
  sizes = repeat(1:5, 8)
  groups = [(e - s + 1):e for (s, e) in zip(sizes, cumsum(sizes))]
  n, p = 100, sum(sizes)
  X = randn(n, p)
  β = zeros(p)
  model = [2, 7, 13, 30]
  for k in model
    β[groups[k]] = randn(sizes[k]) .+ 2 .* sign.(randn(sizes[k]))
  end
  y = X * β + 0.5 * randn(n)
  w = sqrt.(sizes)
  λmax = maximum(norm(X[:, G]' * y / n) / w[k] for (k, G) in enumerate(groups))   # x = 0 is optimal for λ ≥ λmax
  λ = 0.2 * λmax
  g = ProxGroupL2(λ, groups)
  opt = CDOptions(; optTol = 1e-10)

  # largest violation of the optimality conditions at x with gradient grad:
  # grad[G] = -λw x[G] / ‖x[G]‖ for a nonzero group, ‖grad[G]‖ ≤ λw for a zero one
  function kkt_residual(x, grad, λw)
    r = 0.
    for (k, G) in enumerate(groups)
      nx = norm(x[G])
      r = max(r, iszero(nx) ? norm(grad[G]) - λw[k] : norm(grad[G] + λw[k] * x[G] / nx))
    end
    r
  end
  ls_gradient(x) = -X' * (y - X * x) / n
  nonzero_groups(x) = findall(G -> !iszero(x[G]), groups)
  active_groups(x) = sort!([activeCoordinate(x, a) for a in 1:numActive(x)])

  x_ref = Vector(coordinateDescent!(AtomIterate(p, g), CDLeastSquaresLoss(y, X), g, opt))

  @testset "least squares" begin
    @test kkt_residual(x_ref, ls_gradient(x_ref), λ * w) < 1e-8
    @test issubset(model, nonzero_groups(x_ref))

    # with and without the path from λmax, in both orders
    for warmStart in (true, false), randomize in (true, false)
      x, info = AtomIterate(p, g), CDInfo()
      coordinateDescent!(x, CDLeastSquaresLoss(y, X), g, CDOptions(; optTol = 1e-10, warmStart, randomize), info)
      @test info.converged
      @test active_groups(x) == nonzero_groups(Vector(x))     # no zero group stays active
      @test Vector(x) ≈ x_ref atol = 1e-7
    end

    # warm started from a solution, one full pass confirms it
    x, info = AtomIterate(x_ref, g), CDInfo()
    coordinateDescent!(x, CDLeastSquaresLoss(y, X), g, opt, info)
    @test info.iterations == 1
    @test Vector(x) ≈ x_ref atol = 1e-8
  end

  @testset "quadratic loss" begin
    x = AtomIterate(p, g)
    coordinateDescent!(x, CDQuadraticLoss(X'X / n, -X'y / n), g, CDOptions(; optTol = 1e-10, warmStart = false))
    @test Vector(x) ≈ x_ref atol = 1e-7
  end

  @testset "intercept" begin
    # the intercept fit is the fit to the centered data
    y1 = y .+ 3
    Xc, yc = X .- mean(X; dims = 1), y1 .- mean(y1)
    x1, x2 = AtomIterate(p, g), AtomIterate(p, g)
    coordinateDescent!(x1, CDLeastSquaresLoss(y1, X; intercept = true), g, opt)
    coordinateDescent!(x2, CDLeastSquaresLoss(yc, Xc), g, opt)
    @test Vector(x1) ≈ Vector(x2) atol = 1e-7
    @test kkt_residual(Vector(x1), -Xc' * (yc - Xc * Vector(x1)) / n, λ * w) < 1e-8
  end

  @testset "groups of one coordinate give the lasso" begin
    # the same steps, passes and screening as for ProxL1
    singletons = [[i] for i in 1:p]
    for warmStart in (true, false)
      o = CDOptions(; optTol = 1e-10, warmStart, randomize = false)
      xa, xs, ia, is = AtomIterate(p, singletons), SparseIterate(p), CDInfo(), CDInfo()
      coordinateDescent!(xa, CDLeastSquaresLoss(y, X), ProxGroupL2(λ, singletons, ones(p)), o, ia)
      coordinateDescent!(xs, CDLeastSquaresLoss(y, X), ProximalBase.ProxL1(λ), o, is)
      @test Vector(xa) ≈ Vector(xs) atol = 1e-12
      @test ia.iterations == is.iterations
    end
  end

  @testset "zero solution" begin
    # λ ≥ λmax: x stays at zero
    for warmStart in (true, false)
      x = AtomIterate(p, g)
      coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxGroupL2(λmax, groups), CDOptions(; warmStart))
      @test numActive(x) == 0
    end

    # just below λmax, the group that attains λmax enters
    x = AtomIterate(p, g)
    coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxGroupL2(0.99 * λmax, groups), CDOptions(; warmStart = false))
    kmax = argmax([norm(X[:, G]' * y) / w[k] for (k, G) in enumerate(groups)])
    @test active_groups(x) == [kmax]
  end

  @testset "unpenalized group" begin
    # a group with zero weight is not penalized and is left out of λmax
    w0 = copy(w)
    w0[1] = 0
    g0 = ProxGroupL2(λ, groups, w0)
    for warmStart in (true, false)
      x = AtomIterate(p, g0)
      coordinateDescent!(x, CDLeastSquaresLoss(y, X), g0, CDOptions(; optTol = 1e-10, warmStart))
      xv = Vector(x)
      @test 1 in nonzero_groups(xv)
      @test kkt_residual(xv, ls_gradient(xv), λ * w0) < 1e-8
    end
  end

  @testset "zero columns" begin
    # a zero column in a group of the model stays at zero, and a group of zero
    # columns stays zero, though its curvature is zero
    i, k0 = groups[13][1], 5
    X0 = copy(X)
    X0[:, i] .= 0
    X0[:, groups[k0]] .= 0
    for f in (CDLeastSquaresLoss(y, X0), CDQuadraticLoss(X0'X0 / n, -X0'y / n))
      x = AtomIterate(p, g)
      coordinateDescent!(x, f, g, CDOptions(; optTol = 1e-10, warmStart = false))
      xv = Vector(x)
      @test xv[i] == 0
      @test 13 in nonzero_groups(xv)
      @test !(k0 in nonzero_groups(xv))
      @test !any(isnan, xv)
      @test kkt_residual(xv, -X0' * (y - X0 * xv) / n, λ * w) < 1e-8
    end
  end

  @testset "Float32" begin
    X32, y32 = Float32.(X), Float32.(y)
    g32 = ProxGroupL2(Float32(λ), groups)
    x = AtomIterate(Float32, p, g32)
    coordinateDescent!(x, CDLeastSquaresLoss(y32, X32), g32, CDOptions(; optTol = 1e-6, warmStart = false))
    @test eltype(x) == Float32
    @test Vector(x) ≈ x_ref atol = 1e-3
  end

  @testset "dense starting point" begin
    x = zeros(p)
    @test coordinateDescent!(x, CDLeastSquaresLoss(y, X), g, opt) === x
    @test x ≈ x_ref atol = 1e-8
  end

  @testset "caches" begin
    # the curvatures of the groups, the residuals and Ax match their definitions;
    # a curvature is computed only for a group that leaves zero, so the path,
    # which screens the groups, computes few
    Xc = X .- mean(X; dims = 1)
    f = CDLeastSquaresLoss(y, X; intercept = true)
    x = AtomIterate(p, g)
    coordinateDescent!(x, f, g, CDOptions(; optTol = 1e-10, warmStart = false))
    computed = .!isnan.(f.L)
    @test issubset(nonzero_groups(Vector(x)), findall(computed))
    @test count(computed) < length(groups)
    @test f.L[computed] ≈ [eigmax(Symmetric(Xc[:, G]' * Xc[:, G])) / n for G in groups[computed]]
    @test f.r ≈ y .- mean(y) .- Xc * Vector(x)

    A = X'X / n
    f = CDQuadraticLoss(A, -X'y / n)
    x = AtomIterate(p, g)
    coordinateDescent!(x, f, g, CDOptions(; optTol = 1e-10, warmStart = false))
    computed = .!isnan.(f.L)
    @test issubset(nonzero_groups(Vector(x)), findall(computed))
    @test f.L[computed] ≈ [eigmax(Symmetric(A[G, G])) for G in groups[computed]]
    @test f.Ax ≈ A * Vector(x)
  end

  @testset "argument checks" begin
    # the groups of x must be those of g, and x must match the loss
    @test_throws ArgumentError coordinateDescent!(AtomIterate(p, 10), CDLeastSquaresLoss(y, X), g)
    @test_throws DimensionMismatch coordinateDescent!(AtomIterate(p, g), CDLeastSquaresLoss(y, X[:, 1:end-1]), g)
  end
end

end
