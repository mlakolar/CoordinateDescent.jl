module CVTest

using HighDimensionalModels, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# cv against its fold errors errs (K × L, NaN at the levels a fold path did not
# reach), for folds of equal size whose paths stop at different levels before
# the last: cvm and cvse are NaN at the levels no fold reached, and elsewhere
# the mean of the folds that reached the level and its standard error (NaN for
# one fold); λmin and λ1se are among the first Lc levels, which every fold
# reached.
function check_unreached(cv, errs)
  K, L = size(errs)
  Lr = count(j -> any(!isnan, errs[:, j]), 1:L)
  Lc = count(j -> all(!isnan, errs[:, j]), 1:L)
  @test 1 <= Lc < Lr < L
  @test all(j -> all(!isnan, errs[:, j]), 1:Lc) && all(j -> any(!isnan, errs[:, j]), 1:Lr)
  @test all(isnan, cv.cvm[Lr+1:end]) && all(isnan, cv.cvse[Lr+1:end])
  e = [filter(!isnan, errs[:, j]) for j in 1:Lr]
  se = [length(ej) > 1 ? sqrt(sum(abs2, ej .- mean(ej)) / length(ej) / (length(ej) - 1)) : NaN for ej in e]
  @test cv.cvm[1:Lr] ≈ mean.(e)
  @test isequal(isnan.(cv.cvse[1:Lr]), isnan.(se)) && cv.cvse[1:Lr][.!isnan.(se)] ≈ se[.!isnan.(se)]
  jmin = argmin(cv.cvm[1:Lc])
  @test cv.λmin == cv.λpath[jmin]
  @test cv.λ1se == cv.λpath[findfirst(j -> cv.cvm[j] <= cv.cvm[jmin] + cv.cvse[jmin], 1:jmin)]
end

@testset "CVLassoPath and information criteria" begin
  rng = Xoshiro(61)
  n, p = 150, 40
  X = randn(rng, n, p)
  β = zeros(p); β[1:5] .= [1.0, -0.8, 0.6, 0.5, -0.4]
  y = 0.5 .+ X * β .+ randn(rng, n)

  @testset "cross-validation" begin
    folds = repeat(1:5, outer = 30)
    cv = CVLassoPath(X, y; folds, nλ = 30, intercept = true, rng = Xoshiro(1))
    @test cv.folds == folds && length(cv.cvm) == length(cv.cvse) == length(cv.λpath) == 30
    @test cv.λpath == cv.path.λpath && cv.nnz == nnz.(cv.path.βpath)
    # the same as the loop by hand, with the fold sizes as weights
    errs = zeros(5, 30)
    for k in 1:5
      tr, te = folds .!= k, folds .== k
      fp = LassoPath(X[tr, :], y[tr], cv.λpath; intercept = true)
      errs[k, :] = vec(mean(abs2, y[te] .- predict(fp, X[te, :]); dims = 1))
    end
    @test cv.cvm ≈ vec(mean(errs; dims = 1)) rtol = 1e-6
    @test cv.cvse ≈ vec(std(errs; dims = 1, corrected = true)) ./ sqrt(5) rtol = 1e-6
    @test cv.λmin == cv.λpath[argmin(cv.cvm)]
    @test cv.λ1se >= cv.λmin
    j = findfirst(==(cv.λmin), cv.λpath)
    @test cv.cvm[findfirst(==(cv.λ1se), cv.λpath)] <= cv.cvm[j] + cv.cvse[j]
    # coef and predict at λ1se and λmin
    @test coef(cv) == coef(cv.path, cv.λ1se)
    @test coef(cv; λ = :λmin) == coef(cv.path, cv.λmin)
    b = coef(cv)
    @test predict(cv, X) ≈ b[1] .+ X * b[2:end]
    @test_throws ArgumentError coef(cv; λ = :bogus)
    @test coef(cv; λ = cv.λpath[5]) == coef(cv.path, cv.λpath[5])
    # random folds are balanced, and reproducible from the generator, with or
    # without threads
    cv1 = CVLassoPath(X, y; nfolds = 7, nλ = 20, rng = Xoshiro(3))
    cv2 = CVLassoPath(X, y; nfolds = 7, nλ = 20, rng = Xoshiro(3), threads = false)
    @test cv1.folds == cv2.folds && cv1.cvm == cv2.cvm
    @test extrema([count(==(k), cv1.folds) for k in 1:7]) == (21, 22)
    @test_throws ArgumentError CVLassoPath(X, y; nfolds = 1)
    @test_throws ArgumentError CVLassoPath(X, y; folds = fill(1, n))
    @test_throws DimensionMismatch CVLassoPath(X, y; folds = folds[1:end-1])
    # the levels of the fold paths that stop early are left out
    cvs = CVLassoPath(X, y; folds, nλ = 30, max_hat_s = 10)
    @test all(isfinite, cvs.cvm)
    # integer data
    @test CVLassoPath(round.(Int, 10X), round.(Int, 10y); nfolds = 3, nλ = 5, rng = Xoshiro(5)).λpath isa Vector{Float64}
  end

  @testset "Gram form across the n = p boundary" begin
    # The training folds have n < p. Automatic CV must still use the full
    # data's Gram choice: with the same seeds it agrees exactly with true.
    Xq = X[1:40, :]
    yq = y[1:40]
    folds = repeat(1:5, outer = 8)
    for data in (Xq, sparse(Xq)), gram in (:auto, true, false)
      expected = gram === :auto ? !issparse(data) : gram
      cv = CVLassoPath(data, yq; folds, nλ = 12, λminratio = 0.01,
                       intercept = true, gram, rng = Xoshiro(7), threads = false)
      ref = CVLassoPath(data, yq; folds, nλ = 12, λminratio = 0.01,
                        intercept = true, gram = expected, rng = Xoshiro(7))
      @test cv.λpath == ref.λpath
      @test cv.cvm == ref.cvm && cv.cvse == ref.cvse
      @test cv.λmin == ref.λmin && cv.λ1se == ref.λ1se
    end
    @test_throws ArgumentError CVLassoPath(Xq, yq; gram = :invalid)
  end

  @testset "information criteria" begin
    path = LassoPath(X, y; nλ = 25, intercept = true)
    Ŷ = predict(path, X)
    rss = vec(sum(abs2, y .- Ŷ; dims = 1))
    df = nnz.(path.βpath) .+ 1
    @test aic(path, X, y) ≈ n .* log.(rss ./ n) .+ 2 .* df
    @test bic(path, X, y) ≈ n .* log.(rss ./ n) .+ log(n) .* df
    s = nnz.(path.βpath)
    lbin(s) = log(binomial(big(p), big(s)))
    @test ebic(path, X, y; γ = 1) ≈ bic(path, X, y) .+ 2 .* Float64.(lbin.(s))
    @test ebic(path, X, y; γ = 0) ≈ bic(path, X, y)
    # the criteria pick a model near the true one
    k = argmin(bic(path, X, y))
    @test issubset(1:5, findall(!iszero, path.βpath[k]))
    # no intercept: df counts the nonzeros only; df ≥ n gives NaN
    path0 = LassoPath(X[1:30, :], y[1:30]; nλ = 40, λminratio = 1e-4)
    a0 = aic(path0, X[1:30, :], y[1:30])
    @test all(isnan(a0[j]) == (nnz(path0.βpath[j]) >= 30) for j in eachindex(a0))
    @test_throws ArgumentError ebic(path, X, y; γ = 2)
  end

  @testset "summary of the fold errors" begin
    # three folds of equal weight whose paths stop after levels 4, 3 and 2 of
    # five: λmin minimizes cvm among the two levels every fold reached, though
    # the levels fewer folds reached have smaller means
    errs = [0.7 0.5 0.4 0.3 NaN; 0.8 1.0 0.2 NaN NaN; 0.9 0.6 NaN NaN NaN]
    cvm, cvse, jmin, j1se = HighDimensionalModels._cvsummary(errs, ones(3))
    @test cvm[1:4] ≈ [0.8, 0.7, 0.3, 0.3] && isnan(cvm[5])
    @test cvse[1:3] ≈ [sqrt(0.02 / 6), sqrt(0.14 / 6), 0.1] && all(isnan, cvse[4:5])
    @test (jmin, j1se) == (2, 1)
    # a fold whose path reached no level
    @test_throws ArgumentError HighDimensionalModels._cvsummary([1.0 2.0; NaN NaN], ones(2))
  end

  @testset "levels that no fold path reaches" begin
    # Every fold path stops early (max_hat_s) before the last levels of the
    # full path (#144). Uniform data and explicit folds give the same draws on
    # every Julia version.
    rng = Xoshiro(1)
    X = rand(rng, 40, 80) .- 0.5
    y = X[:, 1:5] * ones(5) .+ 0.3 .* (rand(rng, 40) .- 0.5)
    folds, K = repeat(1:5, 8), 5
    cv = CVLassoPath(X, y; folds, max_hat_s = 15, gram = false, rng = Xoshiro(2))
    # the fold errors from fold paths with the seeds that CVLassoPath draws
    seeds = rand(Xoshiro(2), UInt64, K + 1)
    errs = fill(NaN, K, length(cv.λpath))
    for k in 1:K
      train, test = findall(!=(k), folds), findall(==(k), folds)
      fp = LassoPath(X[train, :], y[train], cv.λpath; max_hat_s = 15, gram = false, rng = Xoshiro(seeds[k]))
      errs[k, 1:length(fp.λpath)] = mean(abs2, y[test] .- predict(fp, X[test, :]); dims = 1)
    end
    check_unreached(cv, errs)
  end
end

end
