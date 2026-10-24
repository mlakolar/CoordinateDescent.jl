module SparseGroupTest

using CoordinateDescent, ProximalBase, StatsModels
using LinearAlgebra, Random, SparseArrays, Statistics, Test

soft(v, t) = sign.(v) .* max.(abs.(v) .- t, 0)

# the largest violation of the optimality conditions of the sparse group lasso
# λ(α Σ|tⱼ| + (1 - α) Σ wₖ‖t_Gₖ‖) at t, for the gradient G of the loss in t
function kkt_sgl(G, t, groups, w, λ, α)
  v = 0.0
  for (k, gr) in enumerate(groups)
    tG, gG = t[gr], G[gr]
    if all(iszero, tG)
      v = max(v, norm(soft(gG, λ * α)) - λ * (1 - α) * w[k])
    else
      nG = norm(tG)
      for (i, ti) in enumerate(tG)
        r = ti == 0 ? max(abs(gG[i]) - λ * α, 0.0) : abs(gG[i] + λ * α * sign(ti) + λ * (1 - α) * w[k] * ti / nG)
        v = max(v, r)
      end
    end
  end
  v
end

@testset "the sparse group lasso" begin
  rng = Xoshiro(73)
  n, p = 160, 30
  X = randn(rng, n, p) .* transpose(1 .+ 2 .* rand(rng, p))     # columns on different scales
  β = zeros(p); β[1:3] .= [1.0, -0.8, 0.5]; β[11:12] .= [0.7, -0.6]
  y = 0.5 .+ X * β .+ 0.5 .* randn(rng, n)
  groups = [1:5, 6:10, 11:20, 21:30]
  w = sqrt.(length.(groups))
  xbar = vec(mean(X; dims = 1)); s = vec(sqrt.(mean((X .- transpose(xbar)) .^ 2; dims = 1)))
  Xs = (X .- transpose(xbar)) ./ transpose(s); yc = y .- mean(y)
  gradt(b) = -Xs' * (yc - Xs * (s .* b)) / n            # the gradient in t = s .* β
  tight = CDOptions(; optTol = 1e-12)

  @testset "sparseGroupLasso, alpha = $α" for α in (0.2, 0.5, 0.95)
    λ = 0.05
    sol = sparseGroupLasso(X, y, λ, groups, tight; alpha = α)
    b = Vector(sol.x)
    @test sol.converged && sol.scales ≈ s && sol.alpha == α
    @test kkt_sgl(gradt(b), s .* b, groups, w, λ, α) < 1e-7
    @test sol.intercept ≈ mean(y) - dot(xbar, b)
    @test sol.activegroups == [k for (k, G) in enumerate(groups) if any(!iszero, b[G])]
    @test Vector(coef(sol)) == [sol.intercept; b] && predict(sol, X) ≈ sol.intercept .+ X * b
    @test dof(sol) == nnz(sol.x) + 1 && nobs(sol) == n
    # warmStart = false: along a path of levels from λmax, to the same solution
    @test Vector(sparseGroupLasso(X, y, λ, groups, CDOptions(tight; warmStart = false); alpha = α).x) ≈ b atol = 1e-6
  end

  @testset "the limits: the lasso and the group lasso" begin
    λ = 0.05
    a = sparseGroupLasso(X, y, λ, groups, tight; alpha = 1)
    b = lasso(X, y, λ; intercept = true, standardizeX = true, optTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-6
    c = sparseGroupLasso(X, y, λ, groups, tight; alpha = 0, standardizeX = false)
    d = groupLasso(X, y, λ, groups, tight; standardize = :none)
    @test Vector(c.x) ≈ Vector(d.x) atol = 1e-6
  end

  @testset "SparseGroupLassoPath" begin
    path = SparseGroupLassoPath(X, y, groups, tight; nλ = 20)
    @test path.alpha == 0.95 && path.scales ≈ s && nobs(path) == n
    @test all(iszero, path.βpath[1]) && nnz(path.βpath[end]) > 0
    # λmax: just below it, a solution is not zero
    @test nnz(sparseGroupLasso(X, y, path.λpath[1] * (1 - 1e-3), groups, tight).x) > 0
    for k in eachindex(path.λpath)
      b = Vector(path.βpath[k])
      @test kkt_sgl(gradt(b), s .* b, groups, w, path.λpath[k], 0.95) < 1e-7
      @test path.β0path[k] ≈ mean(y) - dot(xbar, b)
    end
    q = SparseGroupLassoPath(X, y, path.λpath, groups, tight; screening = false)
    @test all(k -> isapprox(Vector(q.βpath[k]), Vector(path.βpath[k]); atol = 1e-6), eachindex(path.λpath))
    B = coef(path)
    @test size(B) == (p + 1, length(path.λpath)) && coef(path, path.λpath[4]) == B[:, 4]
    @test predict(path, X)[:, 6] ≈ path.β0path[6] .+ X * Vector(path.βpath[6])
    f = fit(SparseGroupLassoPath, X, y, groups, tight; nλ = 20, rng = Xoshiro(1))
    g = SparseGroupLassoPath(X, y, groups, tight; nλ = 20, rng = Xoshiro(1))
    @test f.λpath == g.λpath && f.βpath == g.βpath
    @test length(SparseGroupLassoPath(X, y, groups; nλ = 20, max_groups = 1).λpath) < 20
  end

  @testset "formulas: a group per term" begin
    data = (; y, x1 = X[:, 1], x2 = X[:, 2], g = repeat(["a", "b", "c", "d"], n ÷ 4))
    m = sparseGroupLasso(@formula(y ~ x1 + g + x2), data, 0.05)
    @test m.model.groups == [[1], [2, 3, 4], [5]]
    @test coefnames(m) == ["(Intercept)", "x1", "g: b", "g: c", "g: d", "x2"]
    mp = fit(SparseGroupLassoPath, @formula(y ~ x1 + g + x2), data; nλ = 10)
    @test mp.model isa SparseGroupLassoPath && length(mp.model.λpath) == 10
    q = fit(SparseGroupLassoPath, @formula(y ~ x1 + g + x2), data, mp.model.λpath, CDOptions(; optTol = 1e-10);
            groups = [1:1, 2:5])
    @test q.model.groups == [[1], [2, 3, 4, 5]]
  end

  @testset "errors and conversions" begin
    @test sparseGroupLasso(round.(Int, 10X), y, 0.05, groups).converged
    @test_throws ArgumentError sparseGroupLasso(X, y, 0.05, groups; alpha = 1.5)
    @test_throws ArgumentError sparseGroupLasso(X, y, -0.05, groups)
    @test_throws ArgumentError sparseGroupLasso(X, y, 0.05, [1:5, 6:10])                       # columns in no group
    @test_throws DimensionMismatch sparseGroupLasso(X, y, 0.05, groups; groupweights = ones(3))
    @test_throws ArgumentError sparseGroupLasso(X, y, 0.05, groups; groupweights = -ones(4))
    @test_throws DimensionMismatch sparseGroupLasso(X, y[1:end-1], 0.05, groups)
    @test_throws ArgumentError SparseGroupLassoPath(X, y, [0.1, -0.1], groups)
    Xn = copy(X); Xn[1, 1] = NaN
    @test_throws ArgumentError sparseGroupLasso(Xn, y, 0.05, groups)
  end
end

end
