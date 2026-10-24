module FormulaTest

using CoordinateDescent, ProximalBase, StatsModels
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

@testset "formulas (StatsModels extension)" begin
  rng = Xoshiro(41)
  n = 120
  x1, x2, x3 = randn(rng, n), randn(rng, n), rand(rng, n) .+ 0.5
  g = rand(rng, ["a", "b", "c"], n)
  y = 0.5 .+ x1 .- 0.8 .* x2 .+ 0.6 .* (g .== "b") .+ 0.3 .* randn(rng, n)
  data = (; y, x1, x2, x3, g)
  # the model matrix of y ~ x1 + x2 + g without its intercept column: dummy coding with "a" as the base
  X = [x1 x2 (g .== "b") (g .== "c")]

  @testset "LassoPath: the matrix fit, the names, predict" begin
    m = fit(LassoPath, @formula(y ~ x1 + x2 + g), data; nλ = 20, rng = Xoshiro(1))
    p = LassoPath(X, y; nλ = 20, intercept = true, rng = Xoshiro(1))
    @test m isa FormulaFit && m.model isa LassoPath
    @test m.model.λpath == p.λpath && m.model.βpath == p.βpath && m.model.β0path == p.β0path
    @test coefnames(m) == ["(Intercept)", "x1", "x2", "g: b", "g: c"]
    @test responsename(m) == "y"
    @test coef(m) == coef(p) && coef(m, p.λpath[5]) == coef(p, p.λpath[5])
    @test nobs(m) == n
    new = (; x1 = randn(rng, 7), x2 = randn(rng, 7), g = ["c", "a", "a", "b", "c", "b", "a"])
    Xnew = [new.x1 new.x2 (new.g .== "b") (new.g .== "c")]
    @test predict(m, new) ≈ predict(p, Xnew)
    @test occursin("LassoPath", sprint(show, m)) && occursin("formula", sprint(show, MIME("text/plain"), m))
  end

  @testset "the intercept and function terms" begin
    m = fit(LassoPath, @formula(y ~ 0 + x1 + x2), data; nλ = 10, rng = Xoshiro(2))
    p = LassoPath([x1 x2], y; nλ = 10, rng = Xoshiro(2))
    @test !m.model.intercept && coefnames(m) == ["x1", "x2"] && m.model.βpath == p.βpath
    @test_throws ArgumentError fit(LassoPath, @formula(y ~ x1), data; intercept = false)
    s = lasso(@formula(y ~ x1 + log(x3)), data, 0.05)
    @test coefnames(s) == ["(Intercept)", "x1", "log(x3)"]
    @test Vector(coef(s)) ≈ Vector(coef(lasso([x1 log.(x3)], y, 0.05; intercept = true))) atol = 1e-6
  end

  @testset "contrasts, missing values, weights and offsets" begin
    m = lasso(@formula(y ~ x1 + g), data, 0.05; contrasts = Dict(:g => EffectsCoding()))
    E = [x1 (g .== "b") .- (g .== "a") (g .== "c") .- (g .== "a")]
    @test Vector(coef(m)) ≈ Vector(coef(lasso(E, y, 0.05; intercept = true))) atol = 1e-6
    # rows with a missing value in a variable of the model are left out
    x2m = Vector{Union{Missing, Float64}}(x2); x2m[[3, 10, 50]] .= missing
    m = lasso(@formula(y ~ x1 + x2), (; y, x1, x2 = x2m, g), 0.05)
    keep = setdiff(1:n, [3, 10, 50])
    @test nobs(m) == n - 3
    @test Vector(coef(m)) ≈ Vector(coef(lasso([x1 x2][keep, :], y[keep], 0.05; intercept = true))) atol = 1e-6
    # weights and offsets as vectors or as columns of the data
    w = rand(rng, 1:3, n); o = 0.1 .* randn(rng, n)
    a = lasso(@formula(y ~ x1 + x2), (; data..., w, o), 0.05; weights = :w, offset = :o)
    b = lasso(@formula(y ~ x1 + x2), data, 0.05; weights = w, offset = o)
    c = lasso([x1 x2], y, 0.05; intercept = true, weights = w, offset = o)
    @test Vector(coef(a)) ≈ Vector(coef(c)) atol = 1e-6
    @test Vector(coef(b)) ≈ Vector(coef(c)) atol = 1e-6
  end

  @testset "other estimators" begin
    s = sqrtLasso(@formula(y ~ x1 + x2 + g), data, 0.5)
    @test s.model isa CoordinateDescent.LassoSolution && length(coef(s)) == 5
    @test Vector(coef(s)) ≈ Vector(coef(sqrtLasso(X, y, 0.5; intercept = true))) atol = 1e-5
    a = adaptiveLasso(@formula(y ~ x1 + x2 + g), data, 0.05)
    @test coefnames(a) == ["(Intercept)", "x1", "x2", "g: b", "g: c"]

    cv = fit(CVLassoPath, @formula(y ~ x1 + x2 + g), data; nfolds = 5, nλ = 15, rng = Xoshiro(3))
    cvm = CVLassoPath(X, y; nfolds = 5, nλ = 15, intercept = true, rng = Xoshiro(3))
    @test cv.model.cvm == cvm.cvm && coef(cv) == coef(cvm) && coef(cv; λ = :λmin) == coef(cvm; λ = :λmin)
    @test predict(cv, data; λ = :λmin) ≈ predict(cvm, X; λ = :λmin)

    yb = Float64.(rand(rng, n) .< logistic.(x1 .- x2))
    db = (; data..., yb)
    gl = glmLasso(@formula(yb ~ x1 + x2 + g), db, 0.02)
    gm = glmLasso(X, yb, 0.02)
    @test Vector(coef(gl)) ≈ Vector(coef(gm)) atol = 1e-6
    @test predict(gl, db; type = :link) ≈ predict(gm, X; type = :link) atol = 1e-6
    gp = fit(GLMLassoPath, @formula(yb ~ x1 + x2 + g), db; nλ = 10)
    @test gp.model isa GLMLassoPath && size(coef(gp)) == (5, length(gp.model.λpath))
    pp = fit(GLMLassoPath, @formula(counts ~ x1 + x2), (; data..., counts = Float64.(rand(rng, 0:3, n))),
             ProximalBase.Poisson(); nλ = 5)
    @test predict(pp, data) ≈ exp.(predict(pp, data; type = :link))

    # the group lasso: one group per term, or the groups given
    grp = groupLasso(@formula(y ~ x1 + g + x2), data, 0.05)
    @test grp.model.groups == [[1], [2, 3], [4]]
    @test coefnames(grp) == ["(Intercept)", "x1", "g: b", "g: c", "x2"]
    @test groupLasso(@formula(y ~ x1 + g + x2), data, 0.05; groups = [1:1, 2:4]).model.groups == [[1], [2, 3, 4]]
    gpath = fit(GroupLassoPath, @formula(y ~ x1 + g + x2), data; nλ = 8)
    λs = gpath.model.λpath
    gq = fit(GroupLassoPath, @formula(y ~ x1 + g + x2), data, λs, CDOptions(; optTol = 1e-10))
    @test gq.model.λpath == λs && gq.model.groups == [[1], [2, 3], [4]]

    r = rlasso(@formula(y ~ x1 + x2 + g), data)
    @test r.model isa CoordinateDescent.RLassoSolution && predict(r, data) ≈ y .- residuals(r)
  end

  @testset "errors" begin
    @test_throws ArgumentError lasso(@formula(y ~ x1 + nothere), data, 0.1)
    @test_throws ArgumentError lasso(@formula(y + x2 ~ x1), data, 0.1)          # two responses
    @test_throws ArgumentError lasso(@formula(y ~ x1), [1 2; 3 4], 0.1)
    m = lasso(@formula(y ~ x1), data, 0.1)
    @test_throws ArgumentError predict(m, [1 2; 3 4])
    allmissing = (; y, x1 = Vector{Union{Missing, Float64}}(missing, n))
    @test_throws ArgumentError lasso(@formula(y ~ x1), allmissing, 0.1)
  end
end

end
