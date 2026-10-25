module CVGLMTest

using HighDimensionalModels, ProximalBase, StatsModels
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))
log1pexp(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))

@testset "cross-validation of GLM paths, and weights in CVLassoPath" begin
  rng = Xoshiro(61)
  n, p = 240, 15
  X = randn(rng, n, p)
  β = zeros(p); β[1:3] .= [1.0, -0.8, 0.6]
  y = Float64.(rand(rng, n) .< logistic.(0.2 .+ X * β))
  K = 4
  fl = repeat(1:K, n ÷ K)

  # the fold paths as CVGLMLassoPath computes them: the generator rng gives the
  # seeds of the full path (the last) and of each fold
  function foldpaths(X, y, loss, rng; kw...)
    seeds = rand(rng, UInt64, K + 1)
    full = GLMLassoPath(X, y, loss; nλ = 15, rng = Xoshiro(seeds[end]), kw...)
    full, [GLMLassoPath(X[fl .!= k, :], y[fl .!= k], full.λpath, loss; rng = Xoshiro(seeds[k]),
                        (w -> (; weights = w[fl .!= k]))(get(kw, :weights, ones(n)))...,
                        (o -> (; offset = o[fl .!= k]))(get(kw, :offset, zeros(n)))...) for k in 1:K]
  end

  @testset "the deviance by hand, with weights and an offset" begin
    w = rand(Xoshiro(1), 1:3, n) .* 1.0
    o = 0.2 .* randn(Xoshiro(2), n)
    cv = CVGLMLassoPath(X, y; folds = fl, nλ = 15, rng = Xoshiro(7), weights = w, offset = o)
    full, fps = foldpaths(X, y, ProximalBase.Logistic(), Xoshiro(7); weights = w, offset = o)
    @test cv.λpath == full.λpath && cv.measure == :deviance
    L = length(cv.λpath)
    errs = fill(NaN, K, L)
    for k in 1:K
      test = findall(==(k), fl)
      for j in eachindex(fps[k].λpath)
        η = fps[k].β0path[j] .+ X[test, :] * Vector(fps[k].βpath[j]) .+ o[test]
        d = 2 .* (log1pexp.(η) .- y[test] .* η)                 # the saturated loss is zero for y ∈ {0, 1}
        errs[k, j] = sum(w[test] .* d) / sum(w[test])
      end
    end
    wk = [sum(w[fl .== k]) for k in 1:K]
    for j in 1:L
      ok = findall(k -> !isnan(errs[k, j]), 1:K)
      m = sum(wk[ok] .* errs[ok, j]) / sum(wk[ok])
      @test cv.cvm[j] ≈ m
    end
    @test cv.λmin == cv.λpath[argmin(cv.cvm)]
    @test cv.λ1se >= cv.λmin
  end

  @testset "measures, threads, coef and predict" begin
    for measure in (:deviance, :mse, :mae, :class)
      a = CVGLMLassoPath(X, y; measure, nfolds = 4, nλ = 15, rng = Xoshiro(3), threads = true)
      b = CVGLMLassoPath(X, y; measure, nfolds = 4, nλ = 15, rng = Xoshiro(3), threads = false)
      @test a.cvm == b.cvm && a.folds == b.folds && a.λ1se == b.λ1se
      @test all(isfinite, a.cvm) && all(>=(0), a.cvm)
    end
    cv = CVGLMLassoPath(X, y; nfolds = 4, nλ = 15, rng = Xoshiro(3))
    @test coef(cv) == coef(cv.path, cv.λ1se) && coef(cv; λ = :λmin) == coef(cv.path, cv.λmin)
    b = coef(cv; λ = :λmin)
    @test predict(cv, X; λ = :λmin, type = :link) ≈ b[1] .+ X * Vector(b[2:end])
    @test predict(cv, X; λ = :λmin) ≈ logistic.(b[1] .+ X * Vector(b[2:end]))
    o = randn(Xoshiro(4), n)
    @test predict(cv, X; type = :link, offset = o) ≈ predict(cv, X; type = :link) .+ o
    # the minimum is away from λmax on a signal
    @test cv.λmin < cv.λpath[1] && nnz(coef(cv; λ = :λmin)) > 1
    f = fit(CVGLMLassoPath, X, y; nfolds = 4, nλ = 15, rng = Xoshiro(3))
    @test f.cvm == cv.cvm
    # integer data are converted to floating point
    Xi = round.(Int, 10 .* X)
    @test CVGLMLassoPath(Xi, round.(Int, y); nfolds = 4, nλ = 10, rng = Xoshiro(3)).cvm ≈
          CVGLMLassoPath(Float64.(Xi), y; nfolds = 4, nλ = 10, rng = Xoshiro(3)).cvm
    # misclassification goes to zero on separable data
    ys = Float64.(X[:, 1] .> 0)
    cs = CVGLMLassoPath(X, ys; measure = :class, nfolds = 4, nλ = 30, rng = Xoshiro(5), devmax = 1.0)
    @test minimum(cs.cvm) < 0.05
  end

  @testset "Poisson, and the formula method" begin
    counts = Float64.([rand(Xoshiro(i), 0:2) + (X[i, 1] > 0.5) for i in 1:n])
    cv = CVGLMLassoPath(X, counts, ProximalBase.Poisson(); nfolds = 4, nλ = 15, rng = Xoshiro(6))
    @test all(isfinite, cv.cvm) && cv.measure == :deviance
    @test predict(cv, X) ≈ exp.(predict(cv, X; type = :link))
    data = (; y, x1 = X[:, 1], x2 = X[:, 2], x3 = X[:, 3])
    m = fit(CVGLMLassoPath, @formula(y ~ x1 + x2 + x3), data; nfolds = 4, nλ = 10, rng = Xoshiro(8))
    c = CVGLMLassoPath(X[:, 1:3], y; nfolds = 4, nλ = 10, rng = Xoshiro(8))
    @test m.model.cvm ≈ c.cvm && coefnames(m) == ["(Intercept)", "x1", "x2", "x3"]
  end

  @testset "errors" begin
    @test_throws ArgumentError CVGLMLassoPath(X, y; measure = :auc)
    counts = Float64.(rand(Xoshiro(9), 0:3, n))
    @test_throws ArgumentError CVGLMLassoPath(X, counts, ProximalBase.Poisson(); measure = :class)
    @test_throws ArgumentError CVGLMLassoPath(X, 2 .* y .- 1, ProximalBase.SquaredHinge(); measure = :deviance)
    @test_throws ArgumentError CVGLMLassoPath(X, 2 .* y .- 1, ProximalBase.SquaredHinge(); measure = :mse)
    @test_throws DimensionMismatch CVGLMLassoPath(X, y; weights = ones(n - 1))
    @test_throws DimensionMismatch CVGLMLassoPath(X, y; offset = ones(n + 1))
    @test_throws DimensionMismatch CVGLMLassoPath(X, y[1:end-1])
  end

  @testset "CVLassoPath with weights and an offset" begin
    yl = X * β .+ randn(Xoshiro(10), n)
    w = rand(Xoshiro(11), 1:3, n) .* 1.0
    o = 0.3 .* randn(Xoshiro(12), n)
    cv = CVLassoPath(X, yl; folds = fl, nλ = 12, rng = Xoshiro(13), weights = w, offset = o, intercept = true)
    seeds = rand(Xoshiro(13), UInt64, K + 1)
    full = LassoPath(X, yl; nλ = 12, rng = Xoshiro(seeds[end]), weights = w, offset = o, intercept = true)
    @test cv.λpath == full.λpath
    errs = map(1:K) do k
      train, test = fl .!= k, fl .== k
      fp = LassoPath(X[train, :], yl[train], full.λpath; rng = Xoshiro(seeds[k]), weights = w[train],
                     offset = o[train], intercept = true)
      Ŷ = predict(fp, X[test, :])
      [sum(w[test] .* abs2.(yl[test] .- o[test] .- Ŷ[:, j])) / sum(w[test]) for j in axes(Ŷ, 2)]
    end
    wk = [sum(w[fl .== k]) for k in 1:K]
    @test cv.cvm ≈ [sum(wk .* getindex.(errs, j)) / sum(wk) for j in eachindex(full.λpath)]
    # without weights and offset, the folds are weighted by their sizes, as before
    a = CVLassoPath(X, yl; folds = fl, nλ = 12, rng = Xoshiro(13))
    b = CVLassoPath(X, yl; folds = fl, nλ = 12, rng = Xoshiro(13), weights = ones(n))
    @test a.cvm == b.cvm
  end
end

end
