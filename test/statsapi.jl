module StatsAPIEstimatorsTest

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

@testset "StatsAPI methods of the other estimators" begin
  rng = Xoshiro(23)
  n, p = 150, 12
  X = randn(rng, n, p)
  β = zeros(p); β[1:3] .= [1.0, -0.8, 0.6]

  @testset "glmLasso and GLMLassoPath" begin
    y = Float64.(rand(rng, n) .< logistic.(0.3 .+ X * β))
    for intercept in (true, false)
      sol = glmLasso(X, y, 0.02; intercept)
      b = Vector(sol.x)
      @test sol.hasintercept == intercept
      @test coef(sol) isa SparseVector
      @test Vector(coef(sol)) == (intercept ? [sol.intercept; b] : b)
      η = sol.intercept .+ X * b
      @test predict(sol, X; type = :link) ≈ η
      @test predict(sol, X; type = :link) ≈ sol.linearpredictor
      @test predict(sol, X) ≈ logistic.(η)                      # the response scale by default
      @test fitted(sol) ≈ predict(sol, X)
      @test nobs(sol) == n
      @test dof(sol) == count(!iszero, b) + intercept
      @test deviance(sol) == sol.deviance && nulldeviance(sol) == sol.nulldeviance
    end
    sol = glmLasso(X, y, 0.02)
    o = 0.1 .* randn(rng, n)
    @test predict(sol, X; type = :link, offset = o) ≈ predict(sol, X; type = :link) .+ o
    @test_throws ArgumentError predict(sol, X; type = :mean)
    @test_throws DimensionMismatch predict(sol, X[:, 1:end-1])
    @test_throws DimensionMismatch predict(sol, X; offset = o[1:end-1])

    # Poisson: the means are exp(η); the Huber loss: η itself
    counts = Float64.([rand(rng) < 0.5 ? 0 : rand(rng, 0:4) for _ in 1:n])
    sp = glmLasso(X, counts, 0.05, ProximalBase.Poisson())
    @test predict(sp, X) ≈ exp.(predict(sp, X; type = :link))
    sh = glmLasso(X, X * β .+ randn(rng, n), 0.05, ProximalBase.Huber(1.0))
    @test predict(sh, X) ≈ predict(sh, X; type = :link)
    # a loss whose mean is not known has only the link scale
    @test_throws ArgumentError CoordinateDescent._inverselink(ProximalBase.SquaredHinge(), 0.0)

    path = GLMLassoPath(X, y; nλ = 15)
    B = coef(path)
    @test size(B) == (p + 1, length(path.λpath))
    for k in eachindex(path.λpath)
      @test coef(path, path.λpath[k]) == B[:, k]
      @test Vector(B[:, k]) == [path.β0path[k]; Vector(path.βpath[k])]
    end
    λm = (path.λpath[4] + path.λpath[5]) / 2
    @test Vector(coef(path, λm)) ≈ (Vector(B[:, 4]) + Vector(B[:, 5])) / 2
    H = predict(path, X; type = :link)
    @test size(H) == (n, length(path.λpath))
    @test H[:, 6] ≈ path.β0path[6] .+ X * Vector(path.βpath[6])
    @test predict(path, X) ≈ logistic.(H)
    @test nobs(path) == n && nulldeviance(path) == path.nulldeviance
    q = fit(GLMLassoPath, X, y; nλ = 15)
    # the random coordinate orders come from the global generator, so compare to a tolerance
    @test q.λpath == path.λpath && all(k -> isapprox(q.βpath[k], path.βpath[k]; atol = 1e-6), eachindex(q.λpath))
  end

  @testset "groupLasso and GroupLassoPath" begin
    y = 0.5 .+ X * β .+ randn(rng, n)
    groups = [1:3, 4:6, 7:9, 10:12]
    for intercept in (true, false)
      sol = groupLasso(X, y, 0.1, groups; intercept)
      @test Vector(coef(sol)) == (intercept ? [sol.intercept; Vector(sol.x)] : Vector(sol.x))
      @test predict(sol, X) ≈ sol.intercept .+ X * Vector(sol.x)
      @test nobs(sol) == n && dof(sol) == nnz(sol.x) + intercept
    end
    @test_throws DimensionMismatch predict(groupLasso(X, y, 0.1, groups), X[:, 1:5])
    path = GroupLassoPath(X, y, groups; nλ = 10)
    B = coef(path)
    @test size(B) == (p + 1, length(path.λpath))
    @test all(k -> coef(path, path.λpath[k]) == B[:, k], eachindex(path.λpath))
    @test predict(path, X)[:, 5] ≈ path.β0path[5] .+ X * Vector(path.βpath[5])
    @test nobs(path) == n
    q = fit(GroupLassoPath, X, y, groups; nλ = 10)
    @test q.λpath == path.λpath && all(k -> isapprox(q.βpath[k], path.βpath[k]; atol = 1e-6), eachindex(q.λpath))
  end

  @testset "multiLasso and MultiLassoPath" begin
    Y = [X * β X * reverse(β)] .+ randn(rng, n, 2) .+ [1.0 -1.0]
    for intercept in (true, false)
      sol = multiLasso(X, Y, 0.1; intercept)
      C = coef(sol)
      @test C isa SparseMatrixCSC
      @test Matrix(C) == (intercept ? [transpose(sol.intercept); Matrix(sol.B)] : Matrix(sol.B))
      @test predict(sol, X) ≈ transpose(sol.intercept) .+ X * Matrix(sol.B)
      @test nobs(sol) == n && dof(sol) == nnz(sol.B) + 2intercept
    end
    path = MultiLassoPath(X, Y; nλ = 10)
    Cs = coef(path)
    @test length(Cs) == length(path.λpath)
    @test all(k -> coef(path, path.λpath[k]) == Cs[k], eachindex(path.λpath))
    λm = (path.λpath[3] + path.λpath[4]) / 2
    @test Matrix(coef(path, λm)) ≈ (Matrix(Cs[3]) + Matrix(Cs[4])) / 2
    P = predict(path, X)
    @test P[4] ≈ transpose(path.β0path[4]) .+ X * Matrix(path.Bpath[4])
    @test nobs(path) == n
    q = fit(MultiLassoPath, X, Y; nλ = 10)
    @test q.λpath == path.λpath && all(k -> isapprox(q.Bpath[k], path.Bpath[k]; atol = 1e-6), eachindex(q.λpath))
  end

  @testset "rlasso, postLasso and doubleSelection" begin
    y = 0.5 .+ X * β .+ randn(rng, n)
    sol = rlasso(X, y)
    @test Vector(coef(sol)) == [sol.intercept; Vector(sol.x)]
    @test predict(sol, X) ≈ y .- residuals(sol)
    @test nobs(sol) == n && dof(sol) == length(sol.selected) + 1
    @test deviance(sol) ≈ sum(abs2, residuals(sol))
    @test length(coef(rlasso(X, y; intercept = false))) == p

    pl = postLasso(X, y, sol)
    S = pl.selected
    @test Vector(coef(pl)) == [pl.intercept; Vector(pl.x)]
    @test Vector(stderror(pl)) == [pl.se_intercept; Vector(pl.se)]
    V = vcov(pl)
    @test V isa SparseMatrixCSC && size(V) == (p + 1, p + 1)
    @test Matrix(V[[1; S .+ 1], [1; S .+ 1]]) == pl.vcov
    @test sqrt.(diag(Matrix(V))) ≈ Vector(stderror(pl))
    ci = confint(pl)
    z = 1.959963984540054
    @test ci[:, 1] ≈ Vector(coef(pl)) .- z .* Vector(stderror(pl))
    @test ci[:, 2] ≈ Vector(coef(pl)) .+ z .* Vector(stderror(pl))
    @test all(iszero, ci[setdiff(2:p+1, S .+ 1), :])
    @test predict(pl, X) ≈ y .- residuals(pl)
    @test nobs(pl) == n && dof(pl) == length(S) + 1 && deviance(pl) ≈ sum(abs2, residuals(pl))
    pn = postLasso(X, y, sol; intercept = false)
    @test length(coef(pn)) == p && size(vcov(pn)) == (p, p)
    @test Matrix(vcov(pn)[S, S]) == pn.vcov

    d = X[:, 1] .+ randn(rng, n)
    yd = 0.5 .* d .+ X * β .+ randn(rng, n)
    te = doubleSelection(yd, d, X)
    @test coef(te) == [te.estimate] && stderror(te) == [te.se] && vcov(te) == fill(te.se^2, 1, 1)
    @test vec(confint(te)) ≈ collect(te.ci)
    @test confint(te; level = 0.9)[2] - confint(te; level = 0.9)[1] < te.ci[2] - te.ci[1]
    @test nobs(te) == n
    @test_throws ArgumentError confint(te; level = 1.5)
  end

  @testset "fit(CVLassoPath, ...)" begin
    y = X * β .+ randn(rng, n)
    a = fit(CVLassoPath, X, y; nfolds = 5, nλ = 20, rng = Xoshiro(3))
    b = CVLassoPath(X, y; nfolds = 5, nλ = 20, rng = Xoshiro(3))
    @test a.λpath == b.λpath && a.cvm == b.cvm && a.λ1se == b.λ1se
  end
end

end
