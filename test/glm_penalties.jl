module GLMPenaltiesTest

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

# the largest violation of the optimality (stationarity) conditions at t for the
# gradient G of the loss: |Gⱼ| ≤ c at tⱼ = 0, and Gⱼ + dpen(tⱼ) = 0 otherwise
kkt(G, t, c, dpen) = maximum(j -> t[j] == 0 ? max(abs(G[j]) - c[j], 0.0) : abs(G[j] + dpen(t[j])), eachindex(t))
denet(λ, α) = t -> λ * α * sign(t) + λ * (1 - α) * t
dmcp(λ, a) = t -> sign(t) * max(λ - abs(t) / a, 0.0)
dscad(λ, a) = t -> sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))

@testset "penalties in glmLasso and GLMLassoPath" begin
  rng = Xoshiro(53)
  n, p = 250, 15
  X = randn(rng, n, p) .* transpose(1 .+ 2 .* rand(rng, p))     # columns on different scales
  β = zeros(p); β[1:4] .= [0.6, -0.5, 0.4, -0.3]
  y = Float64.(rand(rng, n) .< logistic.(0.3 .+ X * β))
  s = vec(sqrt.(mean((X .- mean(X; dims = 1)) .^ 2; dims = 1)))  # the scales of standardizeX
  Xs = X ./ transpose(s)
  tight = CDOptions(; optTol = 1e-12)
  # the gradient of the mean logistic loss in the coefficients t = s .* β of the standardized columns
  grad(sol) = Xs' * (logistic.(sol.intercept .+ X * Vector(sol.x)) .- y) / n
  λ = 0.02

  @testset "NormL1 as the penalty is the lasso" begin
    a = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; newtonTol = 1e-12)
    b = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; penalty = NormL1(5.0), newtonTol = 1e-12)
    @test Vector(b.x) ≈ Vector(a.x) atol = 1e-8
    @test b.penalty isa NormL1 && b.penalty.λ0 == λ && b.scales === nothing
    @test glmLasso(X, y, λ; alpha = 1).penalty isa NormL1
  end

  @testset "the elastic net, MCP and SCAD on the standardized columns" begin
    for (kw, dpen, c) in (((alpha = 0.5,), denet(λ, 0.5), λ * 0.5),
                          ((penalty = ElasticNet(1, 0.05),), t -> λ * sign(t) + 0.05 * t, λ),
                          ((penalty = MCP(1, 3),), dmcp(λ, 3), λ),
                          ((penalty = SCAD(1, 3.7),), dscad(λ, 3.7), λ))
      sol = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; newtonTol = 1e-12, kw...)
      @test sol.converged && sol.scales ≈ s
      t = s .* Vector(sol.x)
      @test kkt(grad(sol), t, fill(c, p), dpen) < 1e-6
      @test predict(sol, X; type = :link) ≈ sol.linearpredictor       # the coefficients are on the scale of X
    end
    # without standardization the penalty applies to β itself
    sol = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; alpha = 0.5, standardizeX = false, newtonTol = 1e-12)
    @test sol.scales === nothing
    G = X' * (logistic.(sol.intercept .+ X * Vector(sol.x)) .- y) / n
    @test kkt(G, Vector(sol.x), fill(λ * 0.5, p), denet(λ, 0.5)) < 1e-6
    # Poisson
    counts = Float64.([rand(rng, 0:2) + (X[i, 1] > 0) for i in 1:n])
    sp = glmLasso(X, counts, λ, ProximalBase.Poisson(), tight; alpha = 0.5, newtonTol = 1e-12)
    Gp = Xs' * (exp.(sp.intercept .+ X * Vector(sp.x)) .- counts) / n
    @test kkt(Gp, s .* Vector(sp.x), fill(λ * 0.5, p), denet(λ, 0.5)) < 1e-6
  end

  @testset "a nonconvex penalty follows the path from λmax" begin
    a = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; penalty = MCP(1, 3), newtonTol = 1e-12)
    b = glmLasso(X, y, λ, ProximalBase.Logistic(), CDOptions(tight; warmStart = false); penalty = MCP(1, 3),
                 newtonTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-6
    # from zero with numSteps = 0, also a stationary point
    c = glmLasso(X, y, λ, ProximalBase.Logistic(), CDOptions(tight; numSteps = 0); penalty = MCP(1, 3),
                 newtonTol = 1e-12)
    @test kkt(grad(c), s .* Vector(c.x), fill(λ, p), dmcp(λ, 3)) < 1e-6
    # with bounds, on the coefficients of the data
    d = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; penalty = MCP(1, 3), lower = 0, upper = 0.3,
                 newtonTol = 1e-12)
    @test all(0 .<= Vector(d.x) .<= 0.3 + 1e-12) && d.penalty.box.u ≈ 0.3 .* s
  end

  @testset "GLMLassoPath" begin
    lasso_path = GLMLassoPath(X, y; nλ = 15)
    path = GLMLassoPath(X, y, ProximalBase.Logistic(), tight; alpha = 0.5, nλ = 15, newtonTol = 1e-12)
    @test path.λpath[1] ≈ lasso_path.λpath[1] / 0.5
    @test path.penalty isa Scaled && path.scales ≈ s
    @test nnz(path.βpath[1]) == 0 && nnz(path.βpath[end]) > 0
    for k in eachindex(path.λpath)
      l = path.λpath[k]
      b = Vector(path.βpath[k])
      G = Xs' * (logistic.(path.β0path[k] .+ X * b) .- y) / n
      @test kkt(G, s .* b, fill(l * 0.5, p), denet(l, 0.5)) < 1e-6
    end
    @test predict(path, X; type = :link)[:, 7] ≈ path.β0path[7] .+ X * Vector(path.βpath[7])
    mpath = GLMLassoPath(X, y, ProximalBase.Logistic(), tight; penalty = MCP(1, 3), nλ = 15, newtonTol = 1e-12)
    @test nnz(mpath.βpath[1]) == 0
    for k in eachindex(mpath.λpath)
      l = mpath.λpath[k]
      b = Vector(mpath.βpath[k])
      G = Xs' * (logistic.(mpath.β0path[k] .+ X * b) .- y) / n
      @test kkt(G, s .* b, fill(l, p), dmcp(l, 3)) < 1e-6
    end
  end

  @testset "observation weights: replicated rows" begin
    w = rand(Xoshiro(7), 1:3, n)
    rows = reduce(vcat, [fill(i, w[i]) for i in 1:n])
    a = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; weights = w, alpha = 0.5, newtonTol = 1e-12)
    b = glmLasso(X[rows, :], y[rows], λ, ProximalBase.Logistic(), tight; alpha = 0.5, newtonTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-6
    @test a.scales ≈ b.scales
  end

  @testset "the adaptive lasso of a GLM" begin
    first = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; newtonTol = 1e-12)
    a = adaptiveLasso(X, y, λ, ProximalBase.Logistic(), tight; newtonTol = 1e-12)
    ω = [iszero(first.x[j]) ? Inf : 1 / abs(first.x[j]) for j in 1:p]
    b = glmLasso(X, y, λ, ProximalBase.Logistic(), tight; standardizeX = false, penalty = NormL1(1, ω), newtonTol = 1e-12)
    @test a isa CoordinateDescent.GLMLassoSolution
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-6
    @test all(j -> !iszero(first.x[j]) || iszero(a.x[j]), 1:p)        # dropped by the first fit, kept out
    r = adaptiveLasso(X, y, λ, ProximalBase.Logistic(); init = :ridge, initλ = 0.1)
    @test r.converged
    v = adaptiveLasso(X, y, λ, ProximalBase.Logistic(); init = Vector(first.x))
    @test Vector(v.x) ≈ Vector(a.x) atol = 1e-5
    @test adaptiveLasso(round.(Int, 10X), y, λ, ProximalBase.Logistic()).converged
    @test_throws ArgumentError adaptiveLasso(X, y, λ, ProximalBase.Logistic(); init = :ols)
    @test_throws DimensionMismatch adaptiveLasso(X, y, λ, ProximalBase.Logistic(); init = zeros(p - 1))
    @test_throws ArgumentError adaptiveLasso(X, y, -λ, ProximalBase.Logistic())
    @test_throws ArgumentError adaptiveLasso(X, y, λ, ProximalBase.Logistic(); γ = 0)
  end

  @testset "errors" begin
    @test_throws ArgumentError glmLasso(X, y, λ; alpha = 0)
    @test_throws ArgumentError glmLasso(X, y, λ; alpha = 0.5, penalty = MCP(1, 3))
    @test_throws ArgumentError GLMLassoPath(X, y; penalty = Constrained(NormL1(1.0), IndBox(0.0, Inf)))
    Xn = copy(X); Xn[1, 1] = NaN
    @test_throws ArgumentError glmLasso(Xn, y, λ; alpha = 0.5)
  end
end

end
