module GLMLassoTest

using HighDimensionalModels, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

# A custom scalar loss whose curvature query counts model constructions.
struct CountedLogistic <: ProximalBase.ScalarLoss
  models::Base.RefValue{Int}
end
(::CountedLogistic)(η, y) = ProximalBase.Logistic()(η, y)
ProximalBase.deriv(::CountedLogistic, η, y) = ProximalBase.deriv(ProximalBase.Logistic(), η, y)
ProximalBase.deriv2(::CountedLogistic, η, y) = ProximalBase.deriv2(ProximalBase.Logistic(), η, y)
ProximalBase.curvature(loss::CountedLogistic) = (loss.models[] += 1; 0.25)

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

@testset "glm_lasso and GLMLassoPath" begin
  rng = Xoshiro(11)
  n, p = 300, 40
  X = randn(rng, n, p) .* (1 .+ 2 .* rand(rng, p))'          # columns on different scales
  β = zeros(p); β[1:4] .= [1.0, -0.8, 0.6, -0.5]
  y = Float64.(rand(rng, n) .< logistic.(0.3 .+ X * β))
  ω = vec(std(X; dims = 1, corrected = false))

  @testset "logistic" begin
    sol = glm_lasso(X, y, 0.02)
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
    sols = glm_lasso(Xs, y, 0.02)
    @test sols.x[1] ≈ sol.x[1] / 10 rtol = 1e-6
    @test Vector(sols.x)[2:end] ≈ b[2:end] atol = 1e-7
    # without standardization the loadings are one
    sol1 = glm_lasso(X, y, 0.02; standardize_x = false)
    μ1 = logistic.(sol1.intercept .+ X * Vector(sol1.x))
    @test kkt(X' * (μ1 .- y) / n, Vector(sol1.x), 0.02, ones(p)) < 1e-7
    # without an intercept
    sol0 = glm_lasso(X, y, 0.02; intercept = false)
    @test sol0.intercept == 0
    μ0 = logistic.(X * Vector(sol0.x))
    @test kkt(X' * (μ0 .- y) / n, Vector(sol0.x), 0.02, vec(sqrt.(mean(X .^ 2; dims = 1)))) < 1e-7
    # λ = 0 with n > p is the maximum likelihood estimate
    mle = logistic_mle(X[:, 1:10], y)
    sol00 = glm_lasso(X[:, 1:10], y, 0.0; newton_tol = 1e-14)
    @test [sol00.intercept; Vector(sol00.x)] ≈ mle rtol = 1e-6
    # above λmax only the intercept, logit of the mean
    path = GLMLassoPath(X, y; nλ = 3)
    solmax = glm_lasso(X, y, path.λpath[1])
    @test nnz(solmax.x) == 0
    @test solmax.intercept ≈ log(ȳ / (1 - ȳ))
    # the path from λmax (warm_start = false) gives the direct solution; gaptol = 0
    # stops its last level on the change of the objective, as the direct solve
    solw = glm_lasso(X, y, 0.02, ProximalBase.Logistic(), CDOptions(; warm_start = false, num_steps = 10, gaptol = 0))
    @test Vector(solw.x) ≈ b atol = 1e-6
    @test solw.intercept ≈ b0 atol = 1e-6
    # integer responses and Float32 data
    soli = glm_lasso(X, Int.(y), 0.02)
    @test Vector(soli.x) ≈ b atol = 1e-7
    sol32 = glm_lasso(Float32.(X), Float32.(y), 0.02f0, ProximalBase.Logistic(), CDOptions(; opt_tol = 1e-5);
                     newton_tol = 1e-6)
    @test eltype(sol32.x) == Float32
    @test Vector(sol32.x) ≈ b atol = 2e-3
    @test_throws ArgumentError glm_lasso(X, y .+ 1, 0.02)
    @test_throws ArgumentError glm_lasso(X, y, -0.1)
  end

  @testset "Poisson with weights and an offset" begin
    yp = Float64.([rand(rng, 0:k) for k in rand(rng, 0:6, n)])
    w = Float64.(rand(rng, 1:3, n)); o = 0.2 .* randn(rng, n)
    sol = glm_lasso(X, yp, 0.03, ProximalBase.Poisson(); weights = w, offset = o)
    η = sol.intercept .+ X * Vector(sol.x) .+ o
    μ = exp.(η)
    ωw = vec(sqrt.(sum(w .* (X .- (w' * X) ./ sum(w)) .^ 2; dims = 1) ./ sum(w)))
    @test sol.converged
    @test abs(sum(w .* (μ .- yp))) / sum(w) < 1e-9
    @test kkt(X' * (w .* (μ .- yp)) / sum(w), Vector(sol.x), 0.03, ωw) < 1e-7
    @test sol.linearpredictor ≈ η
    # integer weights are replicated observations
    idx = reduce(vcat, [fill(i, Int(w[i])) for i in 1:n])
    solr = glm_lasso(X[idx, :], yp[idx], 0.03, ProximalBase.Poisson(); offset = o[idx])
    @test Vector(solr.x) ≈ Vector(sol.x) atol = 1e-6
    @test solr.intercept ≈ sol.intercept atol = 1e-6
    @test_throws ArgumentError glm_lasso(X, -yp, 0.03, ProximalBase.Poisson())
    @test_throws ArgumentError glm_lasso(X, yp, 0.03, ProximalBase.Poisson(); weights = -w)
  end

  @testset "paths" begin
    # gaptol = 0: each level stops on the change of the objective, for the
    # near-exact comparisons with the direct solves below
    exact = CDOptions(; gaptol = 0)
    path = GLMLassoPath(X, y, ProximalBase.Logistic(), exact; nλ = 30, λminratio = 0.01)
    @test length(path.λpath) == 30 && all(path.converged) && path.intercept
    @test nnz(path.βpath[1]) == 0
    @test issorted(path.devratio)
    @test all(path.kkt_checks .>= 1)
    for k in (10, 30)
      sol = glm_lasso(X, y, path.λpath[k]; newton_tol = 1e-12)
      @test Vector(path.βpath[k]) ≈ Vector(sol.x) atol = 1e-5
      @test path.β0path[k] ≈ sol.intercept atol = 1e-5
      @test path.devratio[k] ≈ 1 - sol.deviance / sol.nulldeviance atol = 1e-6
    end
    # screening gives the solutions of the path without it
    path0 = GLMLassoPath(X, y, path.λpath, ProximalBase.Logistic(), exact; screening = false)
    @test all(norm(path.βpath[k] - path0.βpath[k]) < 1e-6 for k in eachindex(path.λpath))
    @test all(path0.kkt_checks .== 0)
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

  @testset "automatic grid reuses the GLM problem" begin
    loss = CountedLogistic(Ref(0))
    GLMLassoPath(X, y, loss; nλ = 3, devmax = 1)
    @test loss.models[] == 1
    # Reusing the model initialized by lambda_max must preserve the fit,
    # including scales, weights, offsets, constraints and Newton options.
    w, o = rand(rng, n) .+ 0.5, 0.1 .* randn(rng, n)
    for loss in (ProximalBase.Logistic(), ProximalBase.Poisson()),
        data in (X, sparse(X)), alpha in (1.0, 0.7), intercept in (true, false)
      kw = (; weights = w, offset = o, alpha, intercept, lower = -2.0, upper = 2.0,
              screening = false, devmax = 1, newton_tol = 1e-10, newton_max_iter = 80)
      a = GLMLassoPath(data, y, loss; nλ = 5, λminratio = 0.1, rng = Xoshiro(17), kw...)
      b = GLMLassoPath(data, y, a.λpath, loss; rng = Xoshiro(17), kw...)
      @test all(a.converged) && all(b.converged)
      @test coef(a) ≈ coef(b) atol = 1e-8
      @test a.devratio ≈ b.devratio atol = 1e-10
      @test a.nulldeviance ≈ b.nulldeviance
      @test a.scales == b.scales
    end
    @test_throws ArgumentError GLMLassoPath(X, y; bogus = true)
    @test_throws ArgumentError GLMLassoPath(X, y; nλ = 0)
    @test_throws ArgumentError GLMLassoPath(X, y; λminratio = 0)
  end

  @testset "Huber, the squared loss, a constant column, integer data" begin
    # Huber's loss with every residual within δ is the lasso's; the squared
    # loss itself is for lasso
    yc = X * β .+ randn(rng, n)
    hb = glm_lasso(X, yc, 0.05, ProximalBase.Huber(100.0); newton_tol = 1e-12)
    ls = lasso(X, yc, 0.05, CDOptions(; opt_tol = 1e-12, gaptol = 0); intercept = true, standardize_x = true)
    @test Vector(hb.x) ≈ Vector(ls.x) atol = 1e-6
    @test hb.intercept ≈ ls.intercept atol = 1e-6
    @test_throws ArgumentError glm_lasso(X, yc, 0.05, ProximalBase.Squared())
    # with weights and an intercept, a constant column has loading zero and
    # stays at zero
    w = rand(rng, n) .+ 0.5
    a = glm_lasso([X fill(2.0, n)], y, 0.02; weights = w, newton_tol = 1e-12)
    b = glm_lasso(X, y, 0.02; weights = w, newton_tol = 1e-12)
    @test a.x[p + 1] == 0
    @test Vector(a.x)[1:p] ≈ Vector(b.x) atol = 1e-6
    # integer data and a Bool response
    @test GLMLassoPath(round.(Int, X), y .> 0; nλ = 5).λpath isa Vector{Float64}
  end
end

end
