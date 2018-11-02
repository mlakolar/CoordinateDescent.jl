module LassoTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)


##############################################
#
#  Lasso
#
##############################################

@testset "lasso" begin

  @testset "zero" begin
    n = 100
    p = 10

    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    Xy = X' * Y / n

    lambda = maximum(abs.(Xy)) + 0.1
    out = lasso(X, Y, lambda)
    @test out.x == SparseIterate(p)
  end

  @testset "non-zero" begin
    for i=1:NUMBER_REPEAT
      n = 100
      p = 10
      s = 5

      X = randn(n, p)
      Y = X[:,1:s] * ones(s) + 0.1 * randn(n)

      λ = fill(0.3, p)
      beta = lasso(X, Y, 1., λ, CDOptions(;optTol=1e-12))

      f = CDQuadraticLoss(X'X/n, -X'Y/n)
      g = ProximalBase.ProxL1(1., λ)
      x1 = SparseIterate( p )
      coordinateDescent!(x1, f, g, CDOptions(;optTol=1e-12))
      @test beta.x ≈ x1 atol=1e-5

      @test (maximum(abs.(X'*(Y - X*beta.x) / n)) - 0.3) / 0.3 ≈ 0. atol=1e-5
    end
  end

  @testset "different interfaces" begin
    n = 500
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ = 0.1
    x1 = lasso(X, Y, λ)
    x2 = lasso(X, Y, λ, ones(p))

    @test Vector(x1.x) ≈ Vector(x2.x) atol=1e-5
  end

end

@testset "cd lasso" begin

  for i=1:NUMBER_REPEAT
    n = 200
    p = 50
    s = 10

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + 0.1 * randn(n)

    g = ProximalBase.ProxL1(0.2)
    f1 = CDQuadraticLoss(X'X/n, -X'Y/n)
    f2 = CDLeastSquaresLoss(Y, X)

    x1 = SparseIterate(p)
    x2 = SparseIterate(p)
    coordinateDescent!(x1, f1, g, CDOptions(;optTol=1e-12))
    coordinateDescent!(x2, f2, g, CDOptions(;optTol=1e-12))

    @test maximum(abs.(x1 - x2)) ≈ 0. atol=1e-5
    @test (maximum(abs.(X'*(Y - X*x1) / n)) - 0.2) / 0.2 ≈ 0. atol=1e-5
    @test (maximum(abs.(X'*(Y - X*x2) / n)) - 0.2) / 0.2 ≈ 0. atol=1e-5
  end

end


@testset "cd sqrt-lasso" begin

  @testset "kkt" begin
    for i=1:NUMBER_REPEAT
      n = 100
      p = 50
      s = 5

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 2.8
      g = ProximalBase.ProxL1(λ)
      f = CDSqrtLassoLoss(Y, X)

      x1 = SparseIterate(p)
      coordinateDescent!(x1, f, g, CDOptions(;maxIter=5000, optTol=1e-8))

      @test max(0, maximum(abs.(X'*(Y - X*x1) / norm(Y - X*x1))) - λ) / λ  ≈ 0. atol=1e-3
    end
  end

  @testset "interfaces" begin
    for i=1:NUMBER_REPEAT
      n = 500
      p = 500
      s = 50

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      opt1 = CDOptions(;maxIter=5000, optTol=1e-10, warmStart=true, randomize=false)
      opt2 = CDOptions(;maxIter=5000, optTol=1e-10, warmStart=true, randomize=true)
      opt3 = CDOptions(;maxIter=5000, optTol=1e-10, warmStart=false, randomize=false)
      opt4 = CDOptions(;maxIter=5000, optTol=1e-10, warmStart=false, randomize=true)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      λ = 1.5
      g = ProximalBase.ProxL1(λ)
      f = CDSqrtLassoLoss(Y, X)

      coordinateDescent!(x1, f, g, opt1)
      coordinateDescent!(x2, f, g, opt2)
      coordinateDescent!(x3, f, g, opt3)
      coordinateDescent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-4
      @test Vector(x3) ≈ Vector(x2) atol=1e-4
      @test Vector(x4) ≈ Vector(x2) atol=1e-4

      y1 = sqrtLasso(X, Y, λ, opt1, standardizeX=false)
      y2 = sqrtLasso(X, Y, λ, opt2, standardizeX=false)
      y3 = sqrtLasso(X, Y, λ, opt3, standardizeX=false)
      y4 = sqrtLasso(X, Y, λ, opt4, standardizeX=false)

      @test Vector(y1.x) ≈ Vector(x2) atol=1e-4
      @test Vector(y2.x) ≈ Vector(x2) atol=1e-4
      @test Vector(y3.x) ≈ Vector(x2) atol=1e-4
      @test Vector(y4.x) ≈ Vector(x2) atol=1e-4

      z1 = sqrtLasso(X, Y, λ, ones(p), opt1)
      z2 = sqrtLasso(X, Y, λ, ones(p), opt2)
      z3 = sqrtLasso(X, Y, λ, ones(p), opt3)
      z4 = sqrtLasso(X, Y, λ, ones(p), opt4)

      @test Vector(z1.x) ≈ Vector(x2) atol=1e-4
      @test Vector(z2.x) ≈ Vector(x2) atol=1e-4
      @test Vector(z3.x) ≈ Vector(x2) atol=1e-4
      @test Vector(z4.x) ≈ Vector(x2) atol=1e-4

    end
  end

end


@testset "scaled lasso" begin

  for i=1:NUMBER_REPEAT
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ = rand() / 5.

    opt1 = IterLassoOptions(;maxIter=100, optTol=1e-8, optionsCD=CDOptions(;maxIter=5000, optTol=1e-8))
    opt2 = IterLassoOptions(;maxIter=100, optTol=1e-8, initProcedure=:InitStd, σinit=2., optionsCD=CDOptions(;maxIter=5000, optTol=1e-8))

    x1 = SparseIterate(p)
    x2 = SparseIterate(p)
    sol1 = scaledLasso!(x1, X, Y, λ, ones(p), opt1)
    sol2 = scaledLasso!(x2, X, Y, λ, ones(p), opt2)

    σ1 = sol1.σ
    σ2 = sol2.σ

    @test max.((maximum(abs.(X'*(Y - X*x1) / n)) - λ*σ1), 0.) / (σ1*λ) ≈ 0. atol=1e-4
    @test max.((maximum(abs.(X'*(Y - X*x2) / n)) - λ*σ2), 0.) / (σ2*λ) ≈ 0. atol=1e-4
    @test Vector(x1) ≈ Vector(x2) atol=1e-4
  end


end



@testset "lasso path" begin

  @testset "standardizeX = false" begin
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ1 = 0.3
    λ2 = 0.1
    opt = CDOptions(;maxIter=5000, optTol=1e-8)

    x1 = lasso(X, Y, λ1)
    x2 = lasso(X, Y, λ2)

    λpath = [λ1, λ2]
    path = LassoPath(X, Y, λpath, opt; standardizeX=false)

    @test typeof(path) == LassoPath{Float64}
    @test Vector(path.βpath[1]) ≈ Vector(x1.x) atol=1e-5
    @test Vector(path.βpath[2]) ≈ Vector(x2.x) atol=1e-5

    S1 = findall(!iszero, x1.x)
    S2 = findall(!iszero, x2.x)
    rf = refitLassoPath(path, X, Y)

    @test rf[S1] ≈ X[:,S1] \ Y atol=1e-5
    @test rf[S2] ≈ X[:,S2] \ Y atol=1e-5
  end

  @testset "standardizeX = true" begin
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    loadingX = Array{Float64}(undef, p)
    CoordinateDescent._stdX!(loadingX, X)

    λ1 = 0.3
    λ2 = 0.1
    opt = CDOptions(;maxIter=5000, optTol=1e-8)

    x1 = lasso(X, Y, λ1, loadingX, opt)
    x2 = lasso(X, Y, λ2, loadingX, opt)

    λpath = [λ1, λ2]
    path = LassoPath(X, Y, λpath, opt)

    @test typeof(path) == LassoPath{Float64}
    @test Vector(path.βpath[1]) ≈ Vector(x1.x) atol=1e-5
    @test Vector(path.βpath[2]) ≈ Vector(x2.x) atol=1e-5

    S1 = findall(!iszero, x1.x)
    S2 = findall(!iszero, x2.x)
    rf = refitLassoPath(path, X, Y)

    @test rf[S1] ≈ X[:,S1] \ Y atol=1e-5
    @test rf[S2] ≈ X[:,S2] \ Y atol=1e-5
  end


end


end
