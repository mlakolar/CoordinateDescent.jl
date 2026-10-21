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


end
