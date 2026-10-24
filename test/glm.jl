module GLMLassoTest

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

# max over j of the violation of the optimality conditions of the weighted ℓ1
# penalty λ ∑ⱼ ωⱼ |βⱼ| for the gradient G of the smooth part
function kkt(G, β, λ, ω)
  maximum(j -> β[j] == 0 ? max(abs(G[j]) - λ * ω[j], 0.0) : abs(G[j] + λ * ω[j] * sign(β[j])), eachindex(β))
end

# the unpenalized logistic regression with an intercept, by Newton's method
function logistic_mle(X, y)
  Z = [ones(size(X, 1)) X]
  b = zeros(size(Z, 2))
  for _ in 1:50
    μ = logistic.(Z * b)
    b += (Z' * (μ .* (1 .- μ) .* Z)) \ (Z' * (y - μ))
  end
  b
end

@testset "glmLasso and GLMLassoPath" begin
  rng = Xoshiro(11)
  n, p = 300, 40
  X = randn(rng, n, p) .* (1 .+ 2 .* rand(rng, p))'          # columns on different scales
  β = zeros(p); β[1:4] .= [1.0, -0.8, 0.6, -0.5]
  y = Float64.(rand(rng, n) .< logistic.(0.3 .+ X * β))
  ω = vec(std(X; dims = 1, corrected = false))

  @testset "logistic" begin
    sol = glmLasso(X, y, 0.02)
    b0, b = sol.intercept, Vector(sol.x)
    μ = logistic.(b0 .+ X * b)
    @test sol.converged
    @test abs(mean(μ .- y)) < 1e-9                              # the intercept is optimal
    @test kkt(X' * (μ .- y) / n, b, 0.02, ω) < 1e-7
    @test sol.linearpredictor ≈ b0 .+ X * b
    @test sol.deviance ≈ 2 * sum(log1p.(exp.(sol.linearpredictor)) .- y .* sol.linearpredictor)
    ȳ = mean(y)
    @test sol.nulldeviance ≈ -2n * (ȳ * log(ȳ) + (1 - ȳ) * log(1 - ȳ))
    @test sol.penalty.λ0 == 0.02 && sol.penalty.w ≈ ω
    # standardization: rescaling a column rescales its coefficient
    Xs = copy(X); Xs[:, 1] .*= 10
    sols = glmLasso(Xs, y, 0.02)
    @test sols.x[1] ≈ sol.x[1] / 10 rtol = 1e-6
    @test Vector(sols.x)[2:end] ≈ b[2:end] atol = 1e-7
    # without standardization the loadings are one
    sol1 = glmLasso(X, y, 0.02; standardizeX = false)
    μ1 = logistic.(sol1.intercept .+ X * Vector(sol1.x))
    @test kkt(X' * (μ1 .- y) / n, Vector(sol1.x), 0.02, ones(p)) < 1e-7
    # without an intercept
    sol0 = glmLasso(X, y, 0.02; intercept = false)
    @test sol0.intercept == 0
    μ0 = logistic.(X * Vector(sol0.x))
    @test kkt(X' * (μ0 .- y) / n, Vector(sol0.x), 0.02, vec(sqrt.(mean(X .^ 2; dims = 1)))) < 1e-7
    # λ = 0 with n > p is the maximum likelihood estimate
    mle = logistic_mle(X[:, 1:10], y)
    sol00 = glmLasso(X[:, 1:10], y, 0.0; newtonTol = 1e-14)
    @test [sol00.intercept; Vector(sol00.x)] ≈ mle rtol = 1e-6
    # above λmax only the intercept, logit of the mean
    path = GLMLassoPath(X, y; nλ = 3)
    solmax = glmLasso(X, y, path.λpath[1])
    @test nnz(solmax.x) == 0
    @test solmax.intercept ≈ log(ȳ / (1 - ȳ))
    # the path from λmax (warmStart = false) gives the direct solution
    solw = glmLasso(X, y, 0.02, ProximalBase.Logistic(), CDOptions(; warmStart = false, numSteps = 10))
    @test Vector(solw.x) ≈ b atol = 1e-6
    @test solw.intercept ≈ b0 atol = 1e-6
    # integer responses and Float32 data
    soli = glmLasso(X, Int.(y), 0.02)
    @test Vector(soli.x) ≈ b atol = 1e-7
    sol32 = glmLasso(Float32.(X), Float32.(y), 0.02f0, ProximalBase.Logistic(), CDOptions(; optTol = 1e-5);
                     newtonTol = 1e-6)
    @test eltype(sol32.x) == Float32
    @test Vector(sol32.x) ≈ b atol = 2e-3
    @test_throws ArgumentError glmLasso(X, y .+ 1, 0.02)
    @test_throws ArgumentError glmLasso(X, y, -0.1)
  end

  @testset "Poisson with weights and an offset" begin
    yp = Float64.([rand(rng, 0:k) for k in rand(rng, 0:6, n)])
    w = Float64.(rand(rng, 1:3, n)); o = 0.2 .* randn(rng, n)
    sol = glmLasso(X, yp, 0.03, ProximalBase.Poisson(); weights = w, offset = o)
    η = sol.intercept .+ X * Vector(sol.x) .+ o
    μ = exp.(η)
    ωw = vec(sqrt.(sum(w .* (X .- (w' * X) ./ sum(w)) .^ 2; dims = 1) ./ sum(w)))
    @test sol.converged
    @test abs(sum(w .* (μ .- yp))) / sum(w) < 1e-9
    @test kkt(X' * (w .* (μ .- yp)) / sum(w), Vector(sol.x), 0.03, ωw) < 1e-7
    @test sol.linearpredictor ≈ η
    # integer weights are replicated observations
    idx = reduce(vcat, [fill(i, Int(w[i])) for i in 1:n])
    solr = glmLasso(X[idx, :], yp[idx], 0.03, ProximalBase.Poisson(); offset = o[idx])
    @test Vector(solr.x) ≈ Vector(sol.x) atol = 1e-6
    @test solr.intercept ≈ sol.intercept atol = 1e-6
    @test_throws ArgumentError glmLasso(X, -yp, 0.03, ProximalBase.Poisson())
    @test_throws ArgumentError glmLasso(X, yp, 0.03, ProximalBase.Poisson(); weights = -w)
  end

  @testset "paths" begin
    path = GLMLassoPath(X, y; nλ = 30, λminratio = 0.01)
    @test length(path.λpath) == 30 && all(path.converged) && path.intercept
    @test nnz(path.βpath[1]) == 0
    @test issorted(path.devratio)
    @test all(path.kktChecks .>= 1)
    for k in (10, 30)
      sol = glmLasso(X, y, path.λpath[k]; newtonTol = 1e-12)
      @test Vector(path.βpath[k]) ≈ Vector(sol.x) atol = 1e-5
      @test path.β0path[k] ≈ sol.intercept atol = 1e-5
      @test path.devratio[k] ≈ 1 - sol.deviance / sol.nulldeviance atol = 1e-6
    end
    # screening gives the solutions of the path without it
    path0 = GLMLassoPath(X, y, path.λpath; screening = false)
    @test all(norm(path.βpath[k] - path0.βpath[k]) < 1e-6 for k in eachindex(path.λpath))
    @test all(path0.kktChecks .== 0)
    # the deviance stop on separable data, and max_hat_s
    ys = Float64.(X[:, 1] .> 0)
    paths = GLMLassoPath(X, ys; nλ = 100, λminratio = 1e-6)
    @test length(paths.λpath) < 100
    @test last(paths.devratio) >= 0.999
    @test length(GLMLassoPath(X, ys, paths.λpath[1:10]; devmax = 1).λpath) == 10
    pathm = GLMLassoPath(X, y; nλ = 50, max_hat_s = 5)
    @test nnz(last(pathm.βpath)) > 5 && all(nnz(b) <= 5 for b in pathm.βpath[1:end-1])
    # a Poisson path from a range of λ
    yp = Float64.(rand.(rng, [0:k for k in rand(rng, 0:4, n)]))
    pathp = GLMLassoPath(X, yp, range(0.2, 0.02; length = 5), ProximalBase.Poisson())
    @test length(pathp.λpath) == 5 && pathp.loss == ProximalBase.Poisson()
  end

  @testset "Huber, the squared loss, a constant column, integer data" begin
    # Huber's loss with every residual within δ is the lasso's; the squared
    # loss itself is for lasso
    yc = X * β .+ randn(rng, n)
    hb = glmLasso(X, yc, 0.05, ProximalBase.Huber(100.0); newtonTol = 1e-12)
    ls = lasso(X, yc, 0.05, CDOptions(; optTol = 1e-12); intercept = true, standardizeX = true)
    @test Vector(hb.x) ≈ Vector(ls.x) atol = 1e-6
    @test hb.intercept ≈ ls.intercept atol = 1e-6
    @test_throws ArgumentError glmLasso(X, yc, 0.05, ProximalBase.Squared())
    # with weights and an intercept, a constant column has loading zero and
    # stays at zero
    w = rand(rng, n) .+ 0.5
    a = glmLasso([X fill(2.0, n)], y, 0.02; weights = w, newtonTol = 1e-12)
    b = glmLasso(X, y, 0.02; weights = w, newtonTol = 1e-12)
    @test a.x[p + 1] == 0
    @test Vector(a.x)[1:p] ≈ Vector(b.x) atol = 1e-6
    # integer data and a Bool response
    @test GLMLassoPath(round.(Int, X), y .> 0; nλ = 5).λpath isa Vector{Float64}
  end
end

end
