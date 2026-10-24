module LassoTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using ProximalSolvers
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
      g = NormL1(1., λ)
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

    g = NormL1(0.2)
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
      g = NormL1(λ)
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
      g = NormL1(λ)
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
  @test sol.penalty.w ≈ stdX
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
    @test sol.penalty.w ≈ Γ rtol=1e-3

    # KKT for the final penalty
    @test maximum(abs.(X'*r / n) ./ (λ0 * sol.penalty.w)) ≈ 1. atol=1e-6
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

  # one outer iteration meets the stopping rule: the initial σ is the one of
  # the update, not the sample standard deviation of the residuals
  sol3 = @test_logs scaledLasso!(copy(sol.x), X, Y, λ, ones(p), IterLassoOptions(;initProcedure=:WarmStart, maxIter=1, optTol=1e-6, optionsCD=CDOptions(;optTol=1e-12)))
  @test sol3.converged
end

@testset "CDSqrtQuadraticLoss" begin
  n, p, s = 300, 40, 5
  X = randn(n, p) .* (0.5 .+ rand(1, p))
  Y = X[:,1:s] * ones(s) + randn(n)
  λ = 1.2
  tight = CDOptions(;maxIter=10_000, optTol=1e-12)

  # the square-root lasso from the data, with the loss ‖r‖ and the penalty λ
  xd = SparseIterate(p)
  coordinateDescent!(xd, CDSqrtLassoLoss(Y, X), NormL1(λ), tight)

  # from A = X'X/n, b = -X'y/n, c = y'y/n the loss is ‖r‖/√n, so the penalty is λ/√n
  A, b, c = X'X / n, -X'Y / n, dot(Y, Y) / n
  f = CDSqrtQuadraticLoss(A, b, c)
  xg = SparseIterate(p)
  coordinateDescent!(xg, f, NormL1(λ / sqrt(n)), tight)
  @test Vector(xg) ≈ Vector(xd) atol=1e-8
  @test ProximalSolvers.loss_value(f, xg) ≈ norm(Y - X * xg) / sqrt(n)
  grad = zeros(p)
  ProximalBase.gradient!(grad, f, xg)
  @test grad ≈ -X' * (Y - X * xg) / (n * norm(Y - X * xg) / sqrt(n))

  # momentum on and off, the cyclic order with Anderson acceleration, and the
  # path from zero give the same solution
  for o in (CDOptions(;maxIter=10_000, optTol=1e-12, momentum=false),
            CDOptions(;maxIter=10_000, optTol=1e-12, momentum=true),
            CDOptions(;maxIter=10_000, optTol=1e-12, order=:cyclic, anderson=5),
            CDOptions(;maxIter=10_000, optTol=1e-12, warmStart=false))
    x = SparseIterate(p)
    coordinateDescent!(x, CDSqrtQuadraticLoss(A, b, c), NormL1(λ / sqrt(n)), o)
    @test Vector(x) ≈ Vector(xd) atol=1e-8
  end

  # an infinite weight keeps a coordinate at zero; a coordinate the loss does
  # not depend on goes to zero
  ω = ones(p)
  ω[1] = Inf
  x = SparseIterate(p)
  coordinateDescent!(x, CDSqrtQuadraticLoss(A, b, c), NormL1(λ / sqrt(n), ω), tight)
  @test iszero(x[1])
  A0 = copy(A)
  A0[2, :] .= 0
  A0[:, 2] .= 0
  b0 = copy(b)
  b0[2] = 0
  x = SparseIterate(p)
  coordinateDescent!(x, CDSqrtQuadraticLoss(A0, b0, c), NormL1(λ / sqrt(n)), tight)
  @test iszero(x[2])

  @test_throws ArgumentError CDSqrtQuadraticLoss(A, b, Inf)
  @test_throws DimensionMismatch CDSqrtQuadraticLoss(A, b[1:end-1], c)
  @test_throws ArgumentError CDSqrtQuadraticLoss(A + triu(ones(p, p), 1), b, c)
  @test_throws DimensionMismatch CDSqrtQuadraticLoss(A[:, 1:end-1], b[1:end-1], c)
  # without the O(p²) checks of A: the same solution
  x = SparseIterate(p)
  coordinateDescent!(x, CDSqrtQuadraticLoss(A, b, c; check=false), NormL1(λ / sqrt(n)), tight)
  @test Vector(x) ≈ Vector(xd) atol=1e-8
  CDSqrtQuadraticLoss(A + triu(ones(p, p), 1), b, c; check=false)     # not checked
end

@testset "scaled lasso from sufficient statistics" begin
  n, p, s = 400, 60, 6
  X = randn(n, p) .* (0.5 .+ rand(1, p))
  Y = X[:,1:s] * randn(s) + 0.7 * randn(n)
  λ = sqrt(2 * log(p) / n)
  w = vec(sqrt.(sum(abs2, X; dims=1) / n))

  sol = scaledLasso(X, Y, λ, w, IterLassoOptions(;optionsCD=CDOptions(;optTol=1e-12)))
  r = Y - X * sol.x
  @test sol.σ ≈ norm(r) / sqrt(n)
  # the optimality conditions of the lasso at the penalty λσ̂
  c = X' * r / n
  act = findall(!iszero, Vector(sol.x))
  @test maximum(abs.(c) ./ w) ≤ λ * sol.σ * (1 + 1e-8)
  @test c[act] ≈ λ * sol.σ * w[act] .* sign.(Vector(sol.x)[act]) rtol=1e-6

  # from the sufficient statistics, the same solution
  f = CDSqrtQuadraticLoss(X'X / n, -X'Y / n, dot(Y, Y) / n)
  solg = scaledLasso!(SparseIterate(p), f, λ, w, CDOptions(;optTol=1e-12))
  @test Vector(solg.x) ≈ Vector(sol.x) atol=1e-8
  @test solg.σ ≈ sol.σ rtol=1e-8
  @test solg.penalty.λ0 ≈ λ * solg.σ
  @test isempty(solg.residuals)

  # a column of a precision matrix: the scaled lasso of variable j on the
  # others from the covariance matrix, with an infinite weight on j
  S = X'X / n
  j = 3
  ωj = sqrt.(diag(S))
  ωj[j] = Inf
  solj = scaledLasso!(SparseIterate(p), CDSqrtQuadraticLoss(S, -S[:, j], S[j, j]), λ, ωj,
                      CDOptions(;optTol=1e-12))
  @test iszero(solj.x[j])
  others = [1:j-1; j+1:p]
  ref = scaledLasso(X[:, others], X[:, j], λ, sqrt.(diag(S))[others],
                    IterLassoOptions(;optionsCD=CDOptions(;optTol=1e-12)))
  @test Vector(solj.x)[others] ≈ Vector(ref.x) atol=1e-8
  @test solj.σ ≈ ref.σ rtol=1e-8

  # real arguments of other types are converted
  sol32 = scaledLasso!(SparseIterate(p), f, 1 // 10, ones(Int, p))
  @test sol32.penalty.λ0 ≈ sol32.σ / 10
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

@testset "lasso path storage" begin
  # the solutions are stored sparsely: memory grows with the nonzeros, not with p
  n, p = 50, 20_000
  X = randn(n, p)
  Y = X[:,1:3] * ones(3) + 0.1 * randn(n)
  λmax = maximum(abs.(X'Y)) / n
  opt = CDOptions(;optTol=1e-8)
  path = LassoPath(X, Y, λmax .* [0.9, 0.7, 0.5], opt; standardizeX=false)

  @test eltype(path.βpath) == SparseVector{Float64, Int}
  @test all(nnz(β) == count(!iszero, β) > 0 for β in path.βpath)   # no stored zeros
  @test Base.summarysize(path.βpath) < 10_000                       # dense: 1.4 MB
  x = SparseIterate(p)
  coordinateDescent!(x, CDLeastSquaresLoss(Y, X), NormL1(0.5 * λmax), opt)
  @test Vector(path.βpath[end]) ≈ Vector(x) atol=1e-6
end

@testset "intercept" begin
  # fitting an unpenalized intercept equals fitting the centered data
  n, p, s = 200, 30, 5
  X = randn(n, p) .+ 2 .* randn(1, p)          # columns with nonzero means
  Y = 3 .+ X[:,1:s] * ones(s) + randn(n)
  Xc, Yc = X .- mean(X, dims=1), Y .- mean(Y)
  xbar = vec(mean(X, dims=1))
  λ = 0.05
  opt = CDOptions(;optTol=1e-12)

  ref = lasso(Xc, Yc, λ, opt)
  sol = lasso(X, Y, λ, opt; intercept=true)
  @test Vector(sol.x) ≈ Vector(ref.x) atol=1e-8
  @test sol.intercept ≈ mean(Y) - dot(xbar, Vector(sol.x))
  @test sol.residuals ≈ Y .- sol.intercept .- X*sol.x
  @test lasso(X, Y, λ, opt).intercept == 0

  # an offset of y changes only the intercept
  sol2 = lasso(X, Y .+ 100, λ, opt; intercept=true)
  @test Vector(sol2.x) ≈ Vector(sol.x) atol=1e-8
  @test sol2.intercept ≈ sol.intercept + 100

  ω = 0.5 .+ rand(p)
  @test Vector(lasso(X, Y, λ, ω, opt; intercept=true).x) ≈ Vector(lasso(Xc, Yc, λ, ω, opt).x) atol=1e-8
  sd = vec(std(X, dims=1, corrected=false))
  sol = lasso(X, Y, λ, opt; intercept=true, standardizeX=true)
  @test sol.penalty.w ≈ sd
  @test Vector(sol.x) ≈ Vector(lasso(Xc, Yc, λ, sd, opt).x) atol=1e-8

  opt = CDOptions(;optTol=1e-10)
  sol = sqrtLasso(X, Y, 2.0, opt; intercept=true)
  @test Vector(sol.x) ≈ Vector(sqrtLasso(Xc, Yc, 2.0, opt).x) atol=1e-7
  @test sol.intercept ≈ mean(Y) - dot(xbar, Vector(sol.x))
  @test Vector(sqrtLasso(X, Y, 2.0, ω, opt; intercept=true).x) ≈ Vector(sqrtLasso(Xc, Yc, 2.0, ω, opt).x) atol=1e-7

  iopt = IterLassoOptions(;optTol=1e-8, optionsCD=CDOptions(;optTol=1e-12))
  s1 = scaledLasso(X, Y, λ, ones(p), iopt; intercept=true)
  s2 = scaledLasso(Xc, Yc, λ, ones(p), iopt)
  @test Vector(s1.x) ≈ Vector(s2.x) atol=1e-6
  @test s1.σ ≈ s2.σ rtol=1e-6
  @test s1.intercept ≈ mean(Y) - dot(xbar, Vector(s1.x))
  for initProcedure in (:Screening, :InitStd)
    iopt = IterLassoOptions(;optTol=1e-8, initProcedure=initProcedure, optionsCD=CDOptions(;optTol=1e-12))
    f1 = feasibleLasso!(SparseIterate(p), X, Y, λ, iopt; intercept=true)
    f2 = feasibleLasso!(SparseIterate(p), Xc, Yc, λ, iopt)
    @test Vector(f1.x) ≈ Vector(f2.x) atol=1e-6
    @test f1.penalty.w ≈ f2.penalty.w rtol=1e-6
  end
end

@testset "lasso path grid" begin
  n, p = 100, 300
  X = randn(n, p) .+ 1
  Y = 2 .+ X[:,1:5] * ones(5) + randn(n)
  Xc, Yc = X .- mean(X, dims=1), Y .- mean(Y)
  sd = vec(std(X, dims=1, corrected=false))
  opt = CDOptions(;optTol=1e-10)

  path = LassoPath(X, Y, opt; nλ=20, intercept=true)
  @test length(path.λpath) == length(path.βpath) == length(path.β0path) == 20
  @test path.λpath[1] ≈ maximum(abs.(Xc'Yc) ./ (n .* sd)) rtol=1e-6   # the smallest λ with zero solution
  @test path.λpath[end] ≈ 0.01 * path.λpath[1]              # default ratio when n < p
  @test issorted(path.λpath; rev=true)
  @test nnz(path.βpath[1]) == 0
  @test path.β0path[1] ≈ mean(Y)
  @test nnz(path.βpath[end]) > 5
  @test all(path.converged)

  # the same as solving at the grid points, and as the centered data
  path2 = LassoPath(X, Y, path.λpath, opt; intercept=true)
  path3 = LassoPath(Xc, Yc, path.λpath, opt)
  for k in (5, 20)
    @test Vector(path2.βpath[k]) ≈ Vector(path.βpath[k]) atol=1e-6
    @test Vector(path3.βpath[k]) ≈ Vector(path.βpath[k]) atol=1e-6
    @test path.β0path[k] ≈ mean(Y) - dot(vec(mean(X, dims=1)), Vector(path.βpath[k]))
  end
  @test all(iszero, path3.β0path)

  # refits with an intercept put it first
  rf = refitLassoPath(path, X, Y)
  S = findall(!iszero, path.βpath[10])
  @test rf[S] ≈ [ones(n) X[:, S]] \ Y

  # options of the grid
  path = LassoPath(X[:, 1:50], Y; nλ=5, standardizeX=false)
  @test path.λpath[end] ≈ 1e-4 * path.λpath[1]              # n > p
  @test path.λpath[1] ≈ maximum(abs.(X[:, 1:50]'Y)) / n rtol=1e-6
  @test nnz(path.βpath[1]) == 0
  @test length(LassoPath(X, Y; nλ=1).λpath) == 1
  λs = LassoPath(X, Y; nλ=3, λminratio=0.5).λpath
  @test λs ≈ λs[1] .* [1, sqrt(0.5), 0.5]
  @test_throws ArgumentError LassoPath(X, Y; nλ=0)
  @test_throws ArgumentError LassoPath(X, Y; λminratio=2)
  @test LassoPath(Int.(round.(X)), Y; nλ=3) isa LassoPath{Float64}
end

@testset "lasso path screening" begin
  n, p = 100, 2000
  X = randn(n, p)
  Y = X[:,1:10] * ones(10) + randn(n)
  opt = CDOptions(;optTol=1e-10)
  for intercept in (false, true)
    p1 = LassoPath(X, Y, opt; nλ=30, intercept=intercept)
    p2 = LassoPath(X, Y, p1.λpath, opt; intercept=intercept, screening=false)
    @test all(>=(1), p1.kktChecks)
    @test all(iszero, p2.kktChecks)
    @test all(p1.converged)
    @test maximum(maximum(abs, Vector(p1.βpath[k]) - Vector(p2.βpath[k])) for k in eachindex(p1.λpath)) < 1e-6
    @test p1.β0path ≈ p2.β0path atol=1e-6
  end
  # screening needs warm starts
  path = LassoPath(X[:, 1:100], Y, [0.5, 0.3], CDOptions(;warmStart=false))
  @test all(iszero, path.kktChecks)
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

  # a long path, with and without an intercept: every refit is that of \
  n, p = 150, 400
  X = randn(n, p)
  Y = X[:, 1:20] * randn(20) + randn(n)
  for intercept in (false, true)
    path = LassoPath(X, Y; nλ=60, λminratio=0.01, intercept=intercept)
    rf = refitLassoPath(path, X, Y)
    for (S, β) in rf
      @test β ≈ (intercept ? [ones(n) X[:, S]] : X[:, S]) \ Y rtol=1e-8
    end
  end
  # supports with n columns or more, and collinear columns, are refit by \
  Xd = [X[:, 1:3] X[:, 1] randn(n, 2)]
  path = LassoPath(Xd, Y, [0.05, 0.01, 0.001], CDOptions(;optTol=1e-8); standardizeX=false)
  rf = refitLassoPath(path, Xd, Y)
  for (S, β) in rf
    @test β ≈ Xd[:, S] \ Y
  end
  Xw = randn(10, 40)
  Yw = randn(10)
  path = LassoPath(Xw, Yw; nλ=20, λminratio=1e-3)
  rf = refitLassoPath(path, Xw, Yw)
  for (S, β) in rf
    @test β ≈ Xw[:, S] \ Yw
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
  # iteration of feasibleLasso! is checked too
  opt = IterLassoOptions(;maxIter=100, optTol=1e-6, optionsCD=CDOptions(;optTol=1e-10))
  sol = @test_logs scaledLasso(X, Y, λ, ones(p), opt)
  @test sol.converged
  @test sol.iterations >= 2
  sol = @test_logs feasibleLasso!(SparseIterate(p), X, Y, λ, opt)
  @test sol.converged
  @test sol.iterations >= 2

  # scaledLasso solves the square-root lasso in one run of coordinate descent,
  # so the outer maxIter and optTol do not matter
  opt = IterLassoOptions(;maxIter=1, optTol=1e-12)
  sol = @test_logs scaledLasso(X, Y, λ, ones(p), opt)
  @test sol.converged
  sol = @test_logs (:warn, r"feasibleLasso! did not converge") feasibleLasso!(SparseIterate(p), X, Y, λ, opt)
  @test !sol.converged

  # a coordinate descent that stops early makes the whole fit unconverged
  opt = IterLassoOptions(;maxIter=100, optTol=1e-2, optionsCD=CDOptions(;maxIter=1))
  sol = @test_logs (:warn, r"coordinate descent did not converge") match_mode=:any scaledLasso(X, Y, λ, ones(p), opt)
  @test !sol.converged
end

@testset "argument types" begin
  # real data and penalties of any type are converted to a common float type
  n, p, s = 100, 20, 3
  X = randn(n, p)
  Y = X[:,1:s] * ones(s) + randn(n)
  opt = CDOptions(;optTol=1e-12)
  ref = Vector(lasso(X, Y, 0.1, opt).x)

  @test Vector(lasso(X, Y, 1//10, opt).x) ≈ ref
  @test Vector(lasso(X, Y, 0.1, ones(Int, p), opt).x) ≈ ref
  @test eltype(lasso(X, Y, 1, opt).x) == Float64

  X32, Y32 = Float32.(X), Float32.(Y)
  sol = lasso(X32, Y32, 0.1, CDOptions(;optTol=1e-6))  # Float64 λ, Float32 data
  @test eltype(sol.x) == Float32
  sol = lasso(X32, Y, 0.1, opt)                         # mixed data promotes
  @test eltype(sol.x) == Float64
  @test Vector(sol.x) ≈ Vector(lasso(Float64.(X32), Y, 0.1, opt).x)

  Xi, Yi = rand(-3:3, n, p), rand(-5:5, n)              # integer data
  @test Vector(lasso(Xi, Yi, 0.1, opt).x) ≈ Vector(lasso(Float64.(Xi), Float64.(Yi), 0.1, opt).x)

  opt = CDOptions(;optTol=1e-10)
  @test Vector(sqrtLasso(X, Y, 2, opt).x) ≈ Vector(sqrtLasso(X, Y, 2.0, opt).x) atol=1e-8
  @test Vector(sqrtLasso(X, Y, 2, ones(Int, p), opt).x) ≈ Vector(sqrtLasso(X, Y, 2.0, ones(p), opt).x) atol=1e-8

  path = LassoPath(X, Y, [1, 0.5f0], opt; standardizeX=false)
  @test path isa LassoPath{Float64}
  @test path.λpath == [1.0, 0.5]

  iopt = IterLassoOptions(;optTol=1e-6, optionsCD=CDOptions(;optTol=1e-10))
  @test Vector(scaledLasso(X, Y, 1//10, ones(Int, p), iopt).x) ≈ Vector(scaledLasso(X, Y, 0.1, ones(p), iopt).x) atol=1e-6
  sol = scaledLasso!(SparseIterate(p), X32, Y, 0.1, ones(p), iopt)  # the iterate fixes the type
  @test eltype(sol.x) == Float64
  sol = feasibleLasso!(SparseIterate(p), X, Y, 1//10, iopt)
  @test Vector(sol.x) ≈ Vector(feasibleLasso!(SparseIterate(p), X, Y, 0.1, iopt).x) atol=1e-6
end

@testset "input checks" begin
  n, p = 50, 10
  X = randn(n, p)
  Y = X[:, 1] + randn(n)

  # negative or NaN penalties are errors, not fits that grow the coefficients
  @test_throws ArgumentError lasso(X, Y, -0.1)
  @test_throws ArgumentError lasso(X, Y, NaN)
  @test_throws ArgumentError lasso(X, Y, 0.1, [-1.; ones(p - 1)])
  @test_throws ArgumentError sqrtLasso(X, Y, -0.1)
  @test_throws ArgumentError scaledLasso(X, Y, -0.1, ones(p))
  @test_throws ArgumentError feasibleLasso!(SparseIterate(p), X, Y, -0.1)
  @test_throws ArgumentError LassoPath(X, Y, [0.1, -0.1])
  @test lasso(X, Y, 0.0).converged

  # NaN or Inf in the data used to give a zero fit reported as converged
  Ynan = copy(Y); Ynan[3] = NaN
  Xinf = copy(X); Xinf[5, 2] = Inf
  @test_throws ArgumentError lasso(X, Ynan, 0.1)
  @test_throws ArgumentError lasso(Xinf, Y, 0.1)
  @test_throws ArgumentError sqrtLasso(X, Ynan, 0.1)
  @test_throws ArgumentError scaledLasso(X, Ynan, 0.1, ones(p))
  @test_throws ArgumentError LassoPath(Xinf, Y)
  @test_throws ArgumentError CDQuadraticLoss([1. NaN; NaN 1.], zeros(2))

  # invalid options, also when built positionally
  @test_throws ArgumentError CDOptions(;maxIter=-1)
  @test_throws ArgumentError CDOptions(;optTol=-1)
  @test_throws ArgumentError CDOptions(;optTol=NaN)
  @test_throws ArgumentError CDOptions(;numSteps=-1)
  o = CDOptions()
  @test CDOptions(o.maxIter, o.optTol, o.randomize, o.warmStart, o.numSteps, o.relativeTol) == o
  @test_throws ArgumentError CDOptions(-1, o.optTol, o.randomize, o.warmStart, o.numSteps, o.relativeTol)
  @test_throws ArgumentError IterLassoOptions(;maxIter=-1)
  @test_throws ArgumentError IterLassoOptions(;optTol=-1)
  @test_throws ArgumentError IterLassoOptions(;initProcedure=:Bogus)
  @test_throws ArgumentError IterLassoOptions(;sinit=0)        # was a BoundsError in scaledLasso
  @test_throws ArgumentError IterLassoOptions(;σinit=0)
end

@testset "the columns most correlated with y" begin
  # |X'y| = (3, 1, 3, 2, 1): the two largest are columns 1 and 3, the three
  # largest also column 4, and the four largest, with the tie at 1, all
  X = Matrix(1.0I, 5, 5)
  y = [3.0, -1.0, -3.0, 2.0, 1.0]
  find = CoordinateDescent._findLargestCorrelations
  @test find(X, y, 1) == find(X, y, 2) == [true, false, true, false, false]
  @test find(X, y, 3) == [true, false, true, true, false]
  @test find(X, y, 4) == trues(5)
  @test find(X, y, 9) == trues(5)                     # s larger than p: all columns
  @test find(ones(5), X, y, 3) == find(X, y, 3)       # the weighted form
  w = [1.0, 1.0, 1.0, 1.0, 4.0]                       # |w .* y| = (3, 1, 3, 2, 4)
  @test find(w, X, y, 1) == [false, false, false, false, true]
  @test find(w, X, y, 2) == [true, false, true, false, true]
end

@testset "sqrt-lasso near a perfect fit" begin
  # with y on the span of a column, ‖r‖² - (X[:, k]'r)² / ‖X[:, k]‖² is zero
  # up to rounding, which made its square root throw a DomainError
  for i in 1:100
    x = 1 .+ rand(3)
    @test sqrtLasso(reshape(x, 3, 1), 2x, 0.1; standardizeX=false).x[1] ≈ 2
  end
  n, p = 20, 60
  for i in 1:20
    X = randn(n, p)
    Y = X[:, 1:3] * ones(3) + 0.01 * randn(n)
    for λ in (1e-3, 1e-6)
      @test sqrtLasso(X, Y, λ; standardizeX=false).x isa SparseIterate
    end
  end
end

@testset "keyword options" begin
  n, p, s = 100, 30, 4
  X = randn(n, p)
  Y = X[:, 1:s] * ones(s) + randn(n)
  λ = 0.05

  # copies with changes keep every other field
  o = CDOptions(; maxIter = 500, optTol = 1e-9, order = :cyclic, warmStart = false, numSteps = 7,
                relativeTol = false, anderson = 3)
  o2 = CDOptions(o; maxIter = 1)
  @test o2.maxIter == 1
  @test all(getfield(o2, k) == getfield(o, k) for k in fieldnames(CDOptions) if k !== :maxIter)
  @test CDOptions(o) == o
  @test CDOptions(o; anderson = 0, randomize = true).order === :random      # randomize replaces order
  @test CDOptions(CDOptions(; momentum = false); optTol = 1e-3).momentum === false
  @test_throws ArgumentError CDOptions(o; maxiter = 1)                       # not a field
  @test_throws ArgumentError CDOptions(o; momentum = true)                   # momentum needs order = :random
  @test CoordinateDescent._withWarmStart(o) == CDOptions(o; warmStart = true)
  io = IterLassoOptions(; maxIter = 7, optTol = 1e-4, initProcedure = :InitStd, sinit = 2, σinit = 0.5,
                        optionsCD = o)
  io2 = IterLassoOptions(io; σinit = 2.0)
  @test io2.σinit == 2.0
  @test all(getfield(io2, k) == getfield(io, k) for k in fieldnames(IterLassoOptions) if k !== :σinit)
  @test_throws ArgumentError IterLassoOptions(io; optionsCD = o, order = :cyclic)
  @test_throws ArgumentError IterLassoOptions(io; sinit = 0)

  # the estimators take the fields as keywords, which change the options passed
  cyc = CDOptions(; order = :cyclic, optTol = 1e-10)
  same(a, b) = Vector(a.x) == Vector(b.x) && a.iterations == b.iterations && a.converged == b.converged
  @test same(lasso(X, Y, λ; order = :cyclic, optTol = 1e-10), lasso(X, Y, λ, cyc))
  @test same(lasso(X, Y, λ, CDOptions(; order = :cyclic); optTol = 1e-10), lasso(X, Y, λ, cyc))
  @test same(lasso(X, Y, λ, ones(p); order = :cyclic, optTol = 1e-10, intercept = true),
             lasso(X, Y, λ, ones(p), cyc; intercept = true))
  @test same(sqrtLasso(X, Y, 1.0; order = :cyclic, optTol = 1e-10), sqrtLasso(X, Y, 1.0, cyc))
  @test same(sqrtLasso(X, Y, 1.0, ones(p); order = :cyclic, optTol = 1e-10), sqrtLasso(X, Y, 1.0, ones(p), cyc))
  @test (@test_logs (:warn, r"did not converge") lasso(X, Y, λ; maxIter = 1)).iterations == 1
  @test_throws ArgumentError lasso(X, Y, λ; maxiter = 10)                    # a misspelled field
  path1 = LassoPath(X, Y; nλ = 10, order = :cyclic, optTol = 1e-10, intercept = true)
  path2 = LassoPath(X, Y, cyc; nλ = 10, intercept = true)
  @test path1.λpath == path2.λpath && path1.βpath == path2.βpath && path1.iterations == path2.iterations
  path1 = LassoPath(X, Y, path2.λpath; warmStart = false, numSteps = 3, order = :cyclic)
  path2 = LassoPath(X, Y, path2.λpath, CDOptions(; warmStart = false, numSteps = 3, order = :cyclic))
  @test path1.βpath == path2.βpath && path1.iterations == path2.iterations
  iopt = IterLassoOptions(; maxIter = 50, optTol = 1e-6, optionsCD = cyc)
  @test same(scaledLasso(X, Y, λ, ones(p); maxIter = 50, optTol = 1e-6, optionsCD = cyc),
             scaledLasso(X, Y, λ, ones(p), iopt))
  @test same(feasibleLasso!(SparseIterate(p), X, Y, λ; maxIter = 50, optTol = 1e-6, optionsCD = cyc),
             feasibleLasso!(SparseIterate(p), X, Y, λ, iopt))
  @test same(feasibleLasso!(SparseIterate(p), X, Y, λ, iopt; initProcedure = :InitStd),
             feasibleLasso!(SparseIterate(p), X, Y, λ, IterLassoOptions(iopt; initProcedure = :InitStd)))
  @test_throws ArgumentError feasibleLasso!(SparseIterate(p), X, Y, λ; numSteps = 3)   # a field of CDOptions
  f = CDSqrtQuadraticLoss(X'X / n, -X'Y / n, dot(Y, Y) / n)
  @test same(scaledLasso!(SparseIterate(p), f, λ, ones(p); order = :cyclic, optTol = 1e-10),
             scaledLasso!(SparseIterate(p), f, λ, ones(p), cyc))
  x1, x2 = SparseIterate(p), SparseIterate(p)
  coordinateDescent!(x1, CDLeastSquaresLoss(Y, X), NormL1(λ); order = :cyclic, optTol = 1e-10)
  r = solve!(x2, CDLeastSquaresLoss(Y, X), NormL1(λ), CDOptions(; order = :cyclic); optTol = 1e-10)
  @test Vector(x1) == Vector(x2) && r.converged
end

@testset "random number generator" begin
  n, p, s = 100, 60, 4
  X = randn(n, p)
  Y = X[:, 1:s] * ones(s) + randn(n)
  λ = 0.02
  run(rng) = (sol = lasso(X, Y, λ; rng); (Vector(sol.x), sol.iterations))

  # a generator of its own makes the passes reproducible, without reseeding the
  # global one
  @test run(Xoshiro(1)) == run(Xoshiro(1))
  Random.seed!(3); u = rand()
  Random.seed!(3); run(Xoshiro(1)); @test rand() == u
  # the default draws from the global generator, as before
  @test (Random.seed!(5); run(Random.default_rng())) == (Random.seed!(5); sol = lasso(X, Y, λ); (Vector(sol.x), sol.iterations))

  # every estimator passes it on
  path(rng) = (P = LassoPath(X, Y; nλ = 20, rng); (P.βpath, P.iterations))
  @test path(Xoshiro(2)) == path(Xoshiro(2))
  cold(rng) = (P = LassoPath(X, Y, [0.2, 0.05]; warmStart = false, rng); (P.βpath, P.iterations))
  @test cold(Xoshiro(2)) == cold(Xoshiro(2))
  sq(rng) = (sol = sqrtLasso(X, Y, 1.0; rng); (Vector(sol.x), sol.iterations))
  @test sq(Xoshiro(3)) == sq(Xoshiro(3))
  sc(rng) = (sol = scaledLasso(X, Y, λ, ones(p); rng); (Vector(sol.x), sol.iterations))
  @test sc(Xoshiro(4)) == sc(Xoshiro(4))
  fe(rng) = (sol = feasibleLasso!(SparseIterate(p), X, Y, λ; initProcedure = :InitStd, rng); (Vector(sol.x), sol.iterations))
  @test fe(Xoshiro(5)) == fe(Xoshiro(5))
  cdrun(rng) = (x = SparseIterate(p); r = solve!(x, CDLeastSquaresLoss(Y, X), NormL1(λ), CDOptions(); rng); (Vector(x), r.passes))
  @test cdrun(Xoshiro(6)) == cdrun(Xoshiro(6))
  Random.seed!(3); path(Xoshiro(2)); cold(Xoshiro(2)); sq(Xoshiro(3)); sc(Xoshiro(4)); fe(Xoshiro(5)); cdrun(Xoshiro(6))
  @test rand() == u
end

@testset "matrix and vector types" begin
  n, p, s = 60, 25, 4
  rng = Xoshiro(21)
  X = randn(rng, n, p)
  Y = X[:, 1:s] * ones(s) + 0.5 * randn(rng, n)
  λ = 0.05
  opt = CDOptions(; optTol = 1e-12, order = :cyclic)
  close(a, b) = isapprox(Vector(a.x), Vector(b.x); atol = 1e-8) && isapprox(a.intercept, b.intercept; atol = 1e-8)

  # the same solution from every type of the same matrix: a transpose or adjoint
  # (copied once), a SparseMatrixCSC (kept sparse) and a view of some columns
  # (read in place)
  ref = lasso(X, Y, λ, opt; intercept = true, standardizeX = true)
  for Xd in (copy(X')', transpose(copy(transpose(X))), sparse(X), view([X randn(rng, n)], :, 1:p))
    @test close(lasso(Xd, Y, λ, opt; intercept = true, standardizeX = true), ref)
    @test close(lasso(Xd, Y, λ, ones(p), opt), lasso(X, Y, λ, ones(p), opt))
    @test close(sqrtLasso(Xd, Y, 1.0, opt), sqrtLasso(X, Y, 1.0, opt))
    @test close(scaledLasso(Xd, Y, λ, ones(p), IterLassoOptions(; optionsCD = opt); intercept = true),
                scaledLasso(X, Y, λ, ones(p), IterLassoOptions(; optionsCD = opt); intercept = true))
    @test close(feasibleLasso!(SparseIterate(p), Xd, Y, λ, IterLassoOptions(; optionsCD = opt); intercept = true),
                feasibleLasso!(SparseIterate(p), X, Y, λ, IterLassoOptions(; optionsCD = opt); intercept = true))
    P1, P2 = LassoPath(Xd, Y, opt; nλ = 8, intercept = true), LassoPath(X, Y, opt; nλ = 8, intercept = true)
    @test P1.λpath ≈ P2.λpath && all(isapprox.(P1.βpath, P2.βpath; atol = 1e-8))
    R1, R2 = refitLassoPath(P1, Xd, Y), refitLassoPath(P2, X, Y)
    @test keys(R1) == keys(R2) && all(isapprox(R1[S], R2[S]; atol = 1e-8) for S in keys(R1))
  end

  # rows selected by a view: a range of rows is read in place, a vector of rows
  # is copied once; the response may be a view as well
  for rows in (1:2:n, randperm(rng, n)[1:40])
    @test close(lasso(view(X, rows, :), view(Y, rows), λ, opt; intercept = true),
                lasso(X[rows, :], Y[rows], λ, opt; intercept = true))
  end

  # a sparse design stays sparse, also in its loadings, means and refits
  Xs = sprandn(rng, n, p, 0.3)
  Xs[:, 3] .= 2.0                                     # a constant column
  Xs[:, 5] .= 0.0                                     # an empty one, which stores zeros
  Ys = Matrix(Xs)[:, 1:s] * ones(s) + 0.5 * randn(rng, n)
  @test CoordinateDescent._design(Xs) === Xs
  @test CoordinateDescent._design(transpose(sparse(Matrix(Xs)'))) isa SparseMatrixCSC
  @test CoordinateDescent._column_means(Xs) ≈ CoordinateDescent._column_means(Matrix(Xs))
  @test CoordinateDescent._column_means(Xs)[[3, 5]] == [2.0, 0.0]          # exactly, as for dense columns
  for intercept in (false, true)
    @test CoordinateDescent._loadings(Xs, intercept) ≈ CoordinateDescent._loadings(Matrix(Xs), intercept)
  end
  e, xbar = randn(rng, n), CoordinateDescent._column_means(Matrix(Xs))
  @test CoordinateDescent._getLoadings!(zeros(p), Xs, e, xbar) ≈ CoordinateDescent._getLoadings!(zeros(p), Matrix(Xs), e, xbar)
  @test close(lasso(Xs, Ys, λ, opt; intercept = true, standardizeX = true),
              lasso(Matrix(Xs), Ys, λ, opt; intercept = true, standardizeX = true))
  @test close(feasibleLasso!(SparseIterate(p), Xs, Ys, λ, IterLassoOptions(; optionsCD = opt); intercept = true),
              feasibleLasso!(SparseIterate(p), Matrix(Xs), Ys, λ, IterLassoOptions(; optionsCD = opt); intercept = true))

  # integer and Float32 data of other array types are converted as before
  @test close(lasso(transpose(round.(Int, 10 * X)'), Y, λ, opt), lasso(Float64.(round.(Int, 10 * X)), Y, λ, opt))
  @test eltype(lasso(sparse(Float32.(X)), Float32.(Y), λ; optTol = 1e-5).x) == Float32
  # and so are integer data for the refits of a path
  Xi = round.(Int, 10 * X)
  Pi = LassoPath(Float64.(Xi), Y, opt; nλ = 8, intercept = true)
  Ri, Rf = refitLassoPath(Pi, Xi, Y), refitLassoPath(Pi, Float64.(Xi), Y)
  @test keys(Ri) == keys(Rf) && all(Ri[S] == Rf[S] for S in keys(Rf))
end

@testset "StatsAPI" begin
  n, p, s = 80, 30, 4
  rng = Xoshiro(31)
  X = randn(rng, n, p)
  Y = 1.5 .+ X[:, 1:s] * ones(s) + 0.5 * randn(rng, n)
  Xnew = randn(rng, 7, p)
  λ = 0.05

  for intercept in (false, true)
    for sol in (lasso(X, Y, λ; intercept), lasso(X, Y, λ, ones(p); intercept), sqrtLasso(X, Y, 1.0; intercept),
                scaledLasso(X, Y, λ, ones(p); intercept), feasibleLasso!(SparseIterate(p), X, Y, λ; intercept))
      β = Vector(sol.x)
      @test sol.hasintercept == intercept
      @test coef(sol) isa SparseVector
      @test Vector(coef(sol)) == (intercept ? [sol.intercept; β] : β)
      @test predict(sol, Xnew) ≈ sol.intercept .+ Xnew * β
      @test predict(sol, X) ≈ Y - residuals(sol)
      @test residuals(sol) === sol.residuals
      @test nobs(sol) == n
      @test dof(sol) == count(!iszero, β) + intercept
      @test deviance(sol) ≈ sum(abs2, sol.residuals)
      @test_throws DimensionMismatch predict(sol, Xnew[:, 1:p-1])
    end
  end
  # the intercept entry of coef is not stored when it is zero
  sol = lasso(X, [ones(n ÷ 2); -ones(n ÷ 2)], 10.0; intercept = true)
  @test sol.intercept == 0 && nnz(coef(sol)) == 0 && length(coef(sol)) == p + 1

  # a solution from sufficient statistics has no residuals
  sol = scaledLasso!(SparseIterate(p), CDSqrtQuadraticLoss(X'X / n, -X'Y / n, dot(Y, Y) / n), λ, ones(p))
  @test Vector(coef(sol)) == Vector(sol.x) && !sol.hasintercept
  @test predict(sol, Xnew) ≈ Xnew * Vector(sol.x)
  @test dof(sol) == count(!iszero, Vector(sol.x))
  @test_throws ArgumentError residuals(sol)
  @test_throws ArgumentError nobs(sol)
  @test_throws ArgumentError deviance(sol)

  # paths: fit is the constructor; coef stacks the solutions, intercepts first
  for intercept in (false, true)
    path = fit(LassoPath, X, Y, CDOptions(; order = :cyclic); nλ = 15, intercept)
    @test path.βpath == LassoPath(X, Y, CDOptions(; order = :cyclic); nλ = 15, intercept).βpath
    B = coef(path)
    @test size(B) == (p + intercept, length(path.λpath)) && B isa SparseMatrixCSC
    for k in eachindex(path.λpath)
      @test Vector(B[:, k]) == (intercept ? [path.β0path[k]; Vector(path.βpath[k])] : Vector(path.βpath[k]))
      @test coef(path, path.λpath[k]) == B[:, k]                            # exact on the grid
    end
    @test predict(path, Xnew) ≈ Matrix(Xnew * reduce(hcat, path.βpath)) .+ transpose(path.β0path)
    @test nobs(path) == path.nobs == n
    # between two grid values, linear interpolation in λ
    λa, λb = path.λpath[5], path.λpath[6]
    λm = (2λa + λb) / 3
    @test Vector(coef(path, λm)) ≈ (2 * Vector(B[:, 5]) + Vector(B[:, 6])) / 3
    @test_throws ArgumentError coef(path, 2 * path.λpath[1])
    @test_throws ArgumentError coef(path, path.λpath[end] / 2)
  end
  path = fit(LassoPath, X, Y, [0.3, 0.1]; standardizeX = false, warmStart = false)
  @test path.λpath == [0.3, 0.1] && nobs(path) == n
end

@testset "gram" begin
  rng = Xoshiro(41)
  n, p, s = 400, 40, 5
  X = randn(rng, n, p) .+ 0.3 .* randn(rng, 1, p)          # columns with nonzero means
  Y = 2.0 .+ X[:, 1:s] * ones(s) + randn(rng, n)
  λ = 0.02
  opt = CDOptions(; optTol = 1e-10, order = :cyclic)

  # the same loss and stopping rule on the Gram matrix as on the residuals
  for intercept in (false, true), standardizeX in (false, true)
    a = lasso(X, Y, λ, opt; intercept, standardizeX, gram = true)
    b = lasso(X, Y, λ, opt; intercept, standardizeX, gram = false)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-7
    @test a.intercept ≈ b.intercept atol = 1e-7
    @test a.residuals ≈ b.residuals atol = 1e-6
    @test a.σ ≈ b.σ
    @test abs(a.iterations - b.iterations) <= 2
  end
  ω = rand(rng, p) .+ 0.5
  @test Vector(lasso(X, Y, λ, ω, opt; gram = true).x) ≈ Vector(lasso(X, Y, λ, ω, opt; gram = false).x) atol = 1e-7
  for intercept in (false, true), standardizeX in (false, true)
    a = sqrtLasso(X, Y, 1.0, opt; intercept, standardizeX, gram = true)
    b = sqrtLasso(X, Y, 1.0, opt; intercept, standardizeX, gram = false)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-6
    @test a.intercept ≈ b.intercept atol = 1e-6
    @test a.σ ≈ b.σ rtol = 1e-6
    @test a.penalty.λ0 == b.penalty.λ0 == 1.0
  end
  for intercept in (false, true), warmStart in (true, false)
    P1 = LassoPath(X, Y, opt; nλ = 15, intercept, warmStart, gram = true)
    P2 = LassoPath(X, Y, opt; nλ = 15, intercept, warmStart, gram = false)
    @test P1.λpath ≈ P2.λpath
    @test all(isapprox(Vector(P1.βpath[k]), Vector(P2.βpath[k]); atol = 1e-6) for k in eachindex(P2.βpath))
    @test P1.β0path ≈ P2.β0path atol = 1e-6
    @test nobs(P1) == n
  end

  # :auto takes the Gram matrix for a path on a dense X with n ≥ p and
  # p ≤ 2000, the residuals otherwise and at one λ
  @test CoordinateDescent._usegram(:auto, X, true)
  @test CoordinateDescent._usegram(:auto, X[1:p, :], true)                # n = p
  @test !CoordinateDescent._usegram(:auto, X[1:p-1, :], true)             # n < p
  @test !CoordinateDescent._usegram(:auto, X, false)                      # one λ
  @test !CoordinateDescent._usegram(:auto, sparse(X), true)
  @test CoordinateDescent._usegram(:auto, falses(2000, 2000), true)
  @test !CoordinateDescent._usegram(:auto, falses(3000, 2001), true)      # p > 2000
  @test CoordinateDescent._usegram(true, sparse(X), false) && !CoordinateDescent._usegram(false, X, true)
  @test_throws ArgumentError lasso(X, Y, λ; gram = :yes)
  a, b = lasso(X, Y, λ, opt), lasso(X, Y, λ, opt; gram = false)
  @test a.x == b.x && a.iterations == b.iterations
  P1, P2 = LassoPath(X, Y, opt; nλ = 15), LassoPath(X, Y, opt; nλ = 15, gram = true)
  @test P1.βpath == P2.βpath && P1.iterations == P2.iterations

  # the Gram matrix of the centered columns, also from several blocks of rows,
  # from a sparse X and in Float32
  xbar = vec(mean(X; dims = 1))
  ref(X, xbar) = (X .- xbar')' * (X .- xbar') / size(X, 1)
  @test CoordinateDescent._gram(X, xbar) ≈ ref(X, xbar)
  @test CoordinateDescent._gram(X, zeros(p)) ≈ X'X / n
  @test issymmetric(CoordinateDescent._gram(X, xbar))
  Xl = randn(rng, 30_000, 40) .+ 1.0                                      # 2 blocks of 26214 rows
  xl = vec(mean(Xl; dims = 1))
  @test CoordinateDescent._gram(Xl, xl) ≈ ref(Xl, xl)
  @test CoordinateDescent._gram(sparse(X), xbar) ≈ ref(X, xbar)
  @test CoordinateDescent._gram(Float32.(X), Float32.(xbar)) ≈ Float32.(ref(X, xbar)) rtol = 1e-4
  @test CoordinateDescent._gram(big.(X[1:20, 1:3]), big.(xbar[1:3])) ≈ ref(X[1:20, 1:3], xbar[1:3])

  # a sparse X on the Gram matrix, and NaN in the data
  Xs = sprandn(rng, n, p, 0.3)
  @test Vector(lasso(Xs, Y, λ, opt; intercept = true, gram = true).x) ≈
        Vector(lasso(Xs, Y, λ, opt; intercept = true, gram = false).x) atol = 1e-7
  Xn = copy(X); Xn[3, 4] = NaN
  @test_throws ArgumentError lasso(Xn, Y, λ; gram = true)
  Yn = copy(Y); Yn[5] = Inf
  @test_throws ArgumentError lasso(X, Yn, λ; gram = true)
end

@testset "observation weights and offsets" begin
  rng = Xoshiro(51)
  n, p, s = 120, 15, 4
  X = randn(rng, n, p) .+ 0.5 .* randn(rng, 1, p)
  Y = 1.0 .+ X[:, 1:s] * ones(s) + randn(rng, n)
  λ = 0.05
  opt = CDOptions(; optTol = 1e-12, order = :cyclic)
  k = rand(rng, 1:3, n)
  rows = reduce(vcat, [fill(i, k[i]) for i in 1:n])
  close(a, b; atol = 1e-7) = isapprox(Vector(a.x), Vector(b.x); atol) && isapprox(a.intercept, b.intercept; atol)

  # integer weights give the fit of the replicated rows, also standardized and
  # on the Gram matrix
  for intercept in (false, true), gram in (false, true)
    a = lasso(X, Y, λ, opt; intercept, standardizeX = true, gram, weights = k)
    b = lasso(X[rows, :], Y[rows], λ, opt; intercept, standardizeX = true, gram)
    @test close(a, b)
    @test a.weights == k && b.weights === nothing
    @test a.σ ≈ sqrt(sum(abs2, b.residuals) / length(rows))
    @test deviance(a) ≈ deviance(b)
    a = sqrtLasso(X, Y, 1.0, opt; intercept, gram, weights = k)
    b = sqrtLasso(X[rows, :], Y[rows], 1.0, opt; intercept, gram)
    @test close(a, b; atol = 1e-6)
  end
  for intercept in (false, true)
    iopt = IterLassoOptions(; optionsCD = opt)
    a = scaledLasso(X, Y, λ, ones(p), iopt; intercept, weights = k)
    b = scaledLasso(X[rows, :], Y[rows], λ, ones(p), iopt; intercept)
    @test close(a, b; atol = 1e-6) && isapprox(a.σ, b.σ; rtol = 1e-6)
    P1 = LassoPath(X, Y, opt; nλ = 10, intercept, weights = k)
    P2 = LassoPath(X[rows, :], Y[rows], opt; nλ = 10, intercept)
    @test P1.λpath ≈ P2.λpath
    @test all(isapprox(Vector(P1.βpath[j]), Vector(P2.βpath[j]); atol = 1e-6) for j in eachindex(P2.βpath))
    @test P1.β0path ≈ P2.β0path atol = 1e-6
  end

  # unit weights and a zero offset are the same as none: today's fit, exactly
  ref = lasso(X, Y, λ, opt; intercept = true)
  for kw in ((weights = ones(n),), (offset = zeros(n),), (weights = ones(Int, n), offset = zeros(n)))
    a = lasso(X, Y, λ, opt; intercept = true, kw...)
    @test Vector(a.x) == Vector(ref.x) && a.iterations == ref.iterations && a.weights === nothing
  end

  # an offset o gives the fit of y - o, also with weights and on the Gram matrix
  o = randn(rng, n)
  w = rand(rng, n) .+ 0.5
  for gram in (false, true)
    @test close(lasso(X, Y, λ, opt; intercept = true, offset = o, gram), lasso(X, Y .- o, λ, opt; intercept = true, gram))
    a = lasso(X, Y, λ, opt; intercept = true, weights = w, offset = o, gram)
    b = lasso(X, Y .- o, λ, opt; intercept = true, weights = w, gram)
    @test close(a, b)
    @test a.residuals ≈ Y .- o .- a.intercept .- X * Vector(a.x)
    @test predict(a, X) ≈ Y .- o .- a.residuals                 # without the offset
  end
  @test close(sqrtLasso(X, Y, 1.0, opt; offset = o), sqrtLasso(X, Y .- o, 1.0, opt))

  # at λ = 0, the weighted intercept and coefficients of weighted least squares
  a = lasso(X, Y, 0.0, opt; intercept = true, weights = w)
  Z = [ones(n) X]
  @test [a.intercept; Vector(a.x)] ≈ (Z' * (w .* Z)) \ (Z' * (w .* Y)) atol = 1e-6

  # zero weights leave observations out
  wz = copy(w); wz[1:10] .= 0
  @test close(lasso(X, Y, λ, opt; intercept = true, weights = wz),
              lasso(X[11:end, :], Y[11:end], λ, opt; intercept = true, weights = w[11:end]))

  # the weighted loadings and Gram matrix, also of a sparse X
  xw = X' * w / sum(w)
  @test CoordinateDescent._loadings(X, true, w) ≈ sqrt.(vec(sum(w .* (X .- xw') .^ 2; dims = 1)) / sum(w))
  @test CoordinateDescent._loadings(sparse(X), true, w) ≈ CoordinateDescent._loadings(X, true, w)
  Gw = (X .- xw')' * (w .* (X .- xw')) / sum(w)
  @test CoordinateDescent._gram(X, xw, w) ≈ Gw
  @test CoordinateDescent._gram(sparse(X), xw, w) ≈ Gw
  @test CoordinateDescent._column_means(X, w) ≈ xw

  # checks of the weights and the offset
  @test_throws ArgumentError lasso(X, Y, λ; weights = -w)
  @test_throws ArgumentError lasso(X, Y, λ; weights = zeros(n))
  @test_throws DimensionMismatch lasso(X, Y, λ; weights = w[1:end-1])
  @test_throws ArgumentError lasso(X, Y, λ; offset = fill(NaN, n))
  @test_throws DimensionMismatch LassoPath(X, Y; offset = o[1:end-1])
end

end
