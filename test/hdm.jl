module HDMTest

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test
using SpecialFunctions: erfcinv

qnorm(t) = -sqrt(2) * erfcinv(2t)

@testset "rlasso, postLasso and doubleSelection" begin
  @testset "theoryLambda" begin
    n, p = 100, 200
    @test theoryLambda(n, p) ≈ 1.1 * qnorm(1 - 0.1 / log(n) / (2p)) / sqrt(n)
    @test theoryLambda(n, p; c = 0.5, γ = 0.05) ≈ 0.5 * qnorm(1 - 0.05 / (2p)) / sqrt(n)
    @test theoryLambda(n, 2p) > theoryLambda(n, p) > theoryLambda(4n, p)
    @test_throws ArgumentError theoryLambda(n, p; γ = 1.5)
  end

  rng = Xoshiro(51)
  n, p = 300, 100
  X = randn(rng, n, p)
  β = zeros(p); β[1:4] .= [1.0, -0.8, 0.6, 0.5]
  y = 1.0 .+ X * β .+ (0.5 .+ abs.(X[:, 5])) .* randn(rng, n)    # heteroskedastic

  @testset "rlasso" begin
    sol = rlasso(X, y)
    @test issubset(1:4, sol.selected)
    @test sol.post && sol.iterations >= 1
    # the coefficients are least squares on the selected columns, with an intercept
    Z = [ones(n) X[:, sol.selected]]
    b = Z \ y
    @test sol.intercept ≈ b[1] atol = 1e-10
    @test Vector(sol.x)[sol.selected] ≈ b[2:end] atol = 1e-10
    @test sol.residuals ≈ y .- Z * b atol = 1e-10
    @test sol.σ ≈ std(sol.residuals)
    @test sol.λ0 ≈ theoryLambda(n, p)
    # the loadings are those of the residuals
    Xc = X .- mean(X; dims = 1)
    @test sol.loadings ≈ vec(sqrt.(mean((Xc .* sol.residuals) .^ 2; dims = 1)))
    # without the refit, the lasso coefficients and hdm's c = 0.5
    sol0 = rlasso(X, y; post = false)
    @test sol0.λ0 ≈ theoryLambda(n, p; c = 0.5) && !sol0.post
    @test issubset(1:4, sol0.selected)
    # pure noise: nothing selected, the intercept is the mean
    soln = rlasso(X, randn(rng, n) .+ 2)
    @test isempty(soln.selected) || length(soln.selected) <= 2
    # integer data
    @test rlasso(round.(Int, 10X), round.(Int, 10y)).selected isa Vector{Int}
  end

  @testset "postLasso" begin
    sol = lasso(X, y, 0.1; intercept = true)
    pl = postLasso(X, y, sol)
    S = pl.selected
    @test S == findall(!iszero, Vector(sol.x))
    Z = [ones(n) X[:, S]]
    b = Z \ y
    e = y .- Z * b
    @test pl.intercept ≈ b[1] && Vector(pl.x)[S] ≈ b[2:end]
    # HC1 by its definition
    A = inv(Z' * Z)
    V = n / (n - size(Z, 2)) * A * (Z' * Diagonal(e .^ 2) * Z) * A
    @test pl.vcov ≈ V rtol = 1e-8
    @test pl.se_intercept ≈ sqrt(V[1, 1]) && Vector(pl.se)[S] ≈ sqrt.(diag(V)[2:end])
    # from rlasso, or from a vector of columns, and without an intercept
    @test postLasso(X, y, rlasso(X, y)).x ≈ rlasso(X, y).x
    pl0 = postLasso(X, y, [1, 2, 3]; intercept = false)
    @test pl0.intercept == 0 && Vector(pl0.x)[1:3] ≈ X[:, 1:3] \ y
    @test_throws ArgumentError postLasso(X[1:3, :], y[1:3], [1, 2, 3])
  end

  @testset "doubleSelection" begin
    # the treatment depends on the controls that also enter y: selecting on y
    # alone would drop them and bias the estimate
    α = 0.5
    rng2 = Xoshiro(52)
    m, q = 400, 150
    W = randn(rng2, m, q)
    d = W[:, 1:3] * [1.0, 1.0, -1.0] .+ randn(rng2, m)
    yd = α .* d .+ W[:, 1:3] * [1.0, 1.0, -1.0] .+ randn(rng2, m)
    r = doubleSelection(yd, d, W)
    @test r.method === :doubleselection && r.n == m
    @test issubset(1:3, r.selected)
    @test abs(r.estimate - α) < 4 * r.se
    @test r.ci[1] < r.estimate < r.ci[2]
    @test r.ci[2] - r.ci[1] ≈ 2 * qnorm(0.975) * r.se
    @test r.t ≈ r.estimate / r.se && 0 <= r.pvalue <= 1
    # the estimate is least squares of y on d and the selected controls
    Z = [ones(m) d W[:, r.selected]]
    @test r.estimate ≈ (Z \ yd)[2] atol = 1e-10
    # partialling out gives nearly the same estimate here
    rp = doubleSelection(yd, d, W; method = :partialling)
    @test rp.method === :partialling && abs(rp.estimate - r.estimate) < 0.5 * r.se
    # forced controls enter the regression
    rf = doubleSelection(yd, d, W; forced = [100])
    @test 100 in rf.selected
    # a wider interval at a higher level
    r99 = doubleSelection(yd, d, W; level = 0.99)
    @test r99.ci[2] - r99.ci[1] > r.ci[2] - r.ci[1]
    @test_throws ArgumentError doubleSelection(yd, d, W; method = :bogus)
    @test occursin("TreatmentEffect", sprint(show, r))
  end

  @testset "feasibleLasso! at the theory level" begin
    x = SparseIterate(p)
    sol = feasibleLasso!(x, X, y; intercept = true)
    x2 = SparseIterate(p)
    sol2 = feasibleLasso!(x2, X, y, theoryLambda(n, p); intercept = true)
    @test Vector(sol.x) ≈ Vector(sol2.x) atol = 1e-6
    @test sol.penalty.λ0 ≈ theoryLambda(n, p)
    @test_throws ArgumentError feasibleLasso!(SparseIterate(p), X, y; λ0 = :bogus)
  end
end

end
