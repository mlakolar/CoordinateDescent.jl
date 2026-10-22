module LassoTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays
using Statistics

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



@testset "lasso Float32" begin
  n, p, s = 200, 20, 5
  X = randn(Float32, n, p)
  Y = X[:,1:s] * ones(Float32, s) + 0.1f0 * randn(Float32, n)

  sol = lasso(X, Y, 0.1f0, CDOptions(;optTol=1e-6))
  @test eltype(sol.x) == Float32
  @test maximum(abs.(X'*(Y - X*sol.x) / n)) ≈ 0.1f0 atol=1e-4
end

@testset "sqrt-lasso standardizeX" begin
  n, p, s = 200, 50, 5
  X = randn(n, p) .* (0.5 .+ rand(1, p))
  Y = X[:,1:s] * ones(s) + randn(n)

  λ = 1.5
  sol = sqrtLasso(X, Y, λ, CDOptions(;maxIter=5000, optTol=1e-10))
  stdX = vec(sqrt.(sum(abs2, X, dims=1) / n))
  r = Y - X*sol.x
  @test sol.penalty.λ ≈ stdX
  @test maximum(abs.(X'*r) ./ (norm(r) * λ * stdX)) ≈ 1. atol=1e-4
end

@testset "scaledLasso" begin
  n, p, s = 500, 100, 10
  X = randn(n, p)
  Y = X[:,1:s] * randn(s) + randn(n)

  λ = 0.05
  opt = IterLassoOptions(;maxIter=100, optTol=1e-8, optionsCD=CDOptions(;optTol=1e-10))
  x = SparseIterate(p)
  sol1 = scaledLasso!(x, X, Y, λ, ones(p), opt)
  sol2 = scaledLasso(X, Y, λ, ones(p), opt)
  @test Vector(sol2.x) ≈ Vector(sol1.x) atol=1e-6
end

@testset "feasible lasso" begin
  n, p, s = 500, 100, 10
  X = randn(n, p)
  Y = X[:,1:s] * randn(s) + randn(n)

  λ0 = 0.1
  for initProcedure in (:Screening, :InitStd, :WarmStart)
    opt = IterLassoOptions(;maxIter=100, optTol=1e-6, initProcedure=initProcedure,
                           optionsCD=CDOptions(;maxIter=5000, optTol=1e-10))
    sol = feasibleLasso!(SparseIterate(p), X, Y, λ0, opt)

    # the loadings are Γ_j = sqrt(∑ X_ij^2 r_i^2 / n) evaluated at the solution
    r = Y - X*sol.x
    Γ = vec(sqrt.(sum(abs2, X .* r, dims=1) / n))
    @test sol.penalty.λ ≈ Γ rtol=1e-3

    # KKT for the final penalty
    @test maximum(abs.(X'*r / n) ./ (λ0 * sol.penalty.λ)) ≈ 1. atol=1e-6
  end

  @test_throws ArgumentError feasibleLasso!(SparseIterate(p), X, Y, λ0, IterLassoOptions(;initProcedure=:Foo))
  @test_throws ArgumentError scaledLasso!(SparseIterate(p), X, Y, λ0, ones(p), IterLassoOptions(;initProcedure=:Foo))
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

@testset "lasso residuals and σ" begin
  n, p, s = 200, 30, 5
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)

  sol = lasso(X, Y, 0.1, CDOptions(;optTol=1e-10))
  @test sol.residuals ≈ Y - X*sol.x
  @test sol.σ ≈ std(Y - X*sol.x)
  @test sol.penalty.λ0 == 0.1
end

@testset "sqrt-lasso path following" begin
  n, p, s = 200, 50, 5
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)

  λ = 1.5
  sol = sqrtLasso(X, Y, λ, CDOptions(;maxIter=5000, optTol=1e-10, warmStart=false); standardizeX=false)
  r = Y - X*sol.x
  @test maximum(abs.(X'*r)) / norm(r) ≈ λ atol=1e-4
end

@testset "scaled lasso σ and warm start" begin
  n, p, s = 500, 100, 10
  X = randn(n, p)
  Y = X[:,1:s] * randn(s) + 0.5 * randn(n)

  λ = sqrt(2 * log(p) / n)
  opt = IterLassoOptions(;maxIter=100, optTol=1e-10, optionsCD=CDOptions(;optTol=1e-12))
  sol = scaledLasso(X, Y, λ, ones(p), opt)
  r = Y - X*sol.x

  # σ is the noise estimate the penalty is scaled by
  @test sol.σ ≈ norm(r) / sqrt(n)
  @test sol.penalty.λ0 ≈ λ * sol.σ rtol=1e-8
  @test maximum(abs.(X'*r / n)) ≈ sol.penalty.λ0 rtol=1e-6
  @test 0.4 < sol.σ < 0.6

  # starting from the solution, the warm start stops at once at the same point
  x = copy(sol.x)
  sol2 = scaledLasso!(x, X, Y, λ, ones(p), IterLassoOptions(;initProcedure=:WarmStart, maxIter=100, optTol=1e-10, optionsCD=CDOptions(;optTol=1e-12)))
  @test Vector(sol2.x) ≈ Vector(sol.x) atol=1e-6
  @test sol2.σ ≈ sol.σ rtol=1e-6
end

@testset "lasso path early stop" begin
  n, p, s = 200, 100, 20
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)

  λpath = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
  λcopy = copy(λpath)
  path = LassoPath(X, Y, λpath, CDOptions(;optTol=1e-8); max_hat_s=10)

  @test λpath == λcopy                            # the input is not modified
  @test length(path.λpath) == length(path.βpath)
  K = length(path.βpath)
  @test 1 <= K < length(λpath)
  @test path.λpath == λpath[1:K]
  @test nnz(path.βpath[K]) > 10
  @test all(nnz(path.βpath[k]) <= 10 for k=1:K-1)
end

@testset "refit lasso path" begin
  n, p, s = 200, 30, 5
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)

  # repeated λ values give repeated supports, which are refit once
  path = LassoPath(X, Y, [0.5, 0.5, 0.1, 0.1], CDOptions(;optTol=1e-8))
  rf = refitLassoPath(path, X, Y)
  supports = unique([findall(!iszero, β) for β in path.βpath])
  @test length(rf) == length(supports) == 2
  for S in supports
    @test rf[S] ≈ X[:, S] \ Y
  end
end

@testset "convergence reporting" begin
  n, p, s = 200, 50, 5
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)
  λ = 0.05

  # lasso and sqrt-lasso report the passes of coordinate descent
  for sol in (lasso(X, Y, λ), lasso(X, Y, λ, ones(p)), sqrtLasso(X, Y, 0.5), sqrtLasso(X, Y, 0.5, ones(p)))
    @test sol.converged
    @test sol.iterations >= 2
  end

  # too few passes: a warning, and the solution says it did not converge
  sol = @test_logs (:warn, r"did not converge") lasso(X, Y, λ, CDOptions(;maxIter=1))
  @test !sol.converged
  @test sol.iterations == 1

  # LassoPath: one entry per λ, warm starts make later λ values cheap
  λpath = [0.3, 0.1, 0.1]
  path = @test_logs LassoPath(X, Y, λpath)
  @test length(path.iterations) == length(path.converged) == length(λpath)
  @test all(path.converged)
  @test path.iterations[3] <= 2                   # restarted at its own solution
  path = @test_logs (:warn, r"did not converge") match_mode=:any LassoPath(X, Y, λpath, CDOptions(;maxIter=1))
  @test !any(path.converged)

  # scaled and feasible lasso: inner passes are summed, and the outer
  # iteration is checked too
  opt = IterLassoOptions(;maxIter=100, optTol=1e-6, optionsCD=CDOptions(;optTol=1e-10))
  sol = @test_logs scaledLasso(X, Y, λ, ones(p), opt)
  @test sol.converged
  @test sol.iterations >= 2
  sol = @test_logs feasibleLasso!(SparseIterate(p), X, Y, λ, opt)
  @test sol.converged
  @test sol.iterations >= 2

  opt = IterLassoOptions(;maxIter=1, optTol=1e-12)
  sol = @test_logs (:warn, r"scaledLasso! did not converge") scaledLasso(X, Y, λ, ones(p), opt)
  @test !sol.converged
  sol = @test_logs (:warn, r"feasibleLasso! did not converge") feasibleLasso!(SparseIterate(p), X, Y, λ, opt)
  @test !sol.converged

  # a coordinate descent that stops early makes the whole fit unconverged
  opt = IterLassoOptions(;maxIter=100, optTol=1e-2, optionsCD=CDOptions(;maxIter=1))
  sol = @test_logs (:warn, r"coordinate descent did not converge") match_mode=:any scaledLasso(X, Y, λ, ones(p), opt)
  @test !sol.converged
end

end
