module CoordinateDescentTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

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

  struct IncompleteLoss <: CoordinateDifferentiableFunction end
  @test_throws ErrorException CoordinateDescent.initialize!(IncompleteLoss(), SparseIterate(2))
  @test_throws ErrorException CoordinateDescent.gradient(IncompleteLoss(), SparseIterate(2), 1)
  @test_throws ErrorException CoordinateDescent.numCoordinates(IncompleteLoss())
  @test_throws ErrorException CoordinateDescent.descendCoordinate!(IncompleteLoss(), ProxZero(), SparseIterate(2), 1)
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
    @test info.maxChange < 1e-10
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
    @test info.maxChange > 1e-10
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
end

end
