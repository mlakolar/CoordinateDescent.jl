module BoundsTest

using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Random, SparseArrays, Statistics, Test

logistic(z) = 1 / (1 + exp(-z))

# the largest violation of 0 ∈ g + ∂pen(b) + N(b), N(b) the normal cone of the
# box l ≤ b ≤ u: ∂pen(bⱼ) is dpen(j, bⱼ) at bⱼ ≠ 0 and [-cⱼ, cⱼ] at zero, and N
# adds (-∞, 0] at a lower bound and [0, ∞) at an upper bound (within tol)
function kkt_box(g, b, c, l, u, dpen; tol = 1e-10)
  v = 0.0
  for j in eachindex(b)
    a, d = b[j] == 0 ? (-c[j], c[j]) : (dpen(j, b[j]), dpen(j, b[j]))
    lo = b[j] <= l[j] + tol ? -Inf : a
    hi = b[j] >= u[j] - tol ? Inf : d
    t = -g[j]
    v = max(v, t < lo ? lo - t : t > hi ? t - hi : 0.0)
  end
  v
end
l1(c) = (j, t) -> c[j] * sign(t)
inbox(b, l, u; tol = 0.0) = all(l .- tol .<= b .<= u .+ tol)
expand(b, p) = b isa Real ? fill(Float64(b), p) : b

@testset "bounds on the coefficients" begin
  rng = Xoshiro(91)
  n, p = 120, 30
  X = randn(rng, n, p) .* transpose(1 .+ 2 .* rand(rng, p))     # columns on different scales
  β = zeros(p); β[1:6] .= [1.0, -0.8, 0.6, 0.5, -0.4, 0.3]
  y = 0.5 .+ X * β .+ randn(rng, n)
  xbar = vec(mean(X; dims = 1))
  Xc = X .- transpose(xbar)
  yc = y .- mean(y)
  ω = vec(sqrt.(mean(Xc .^ 2; dims = 1)))       # the loadings of standardizeX
  λ = 0.05
  # one-sided, two-sided and binding, and per coefficient: some fixed at zero
  lv, uv = fill(-0.3, p), fill(0.7, p)
  lv[3] = 0; lv[2] = uv[2] = 0; uv[5] = 0
  boxes = [(0, Inf), (-Inf, 0), (-0.3, 0.7), (lv, uv)]

  @testset "lasso, lower = $lower, upper = $upper" for (lower, upper) in boxes
    l, u = expand(lower, p), expand(upper, p)
    for intercept in (true, false)
      Z, z = intercept ? (Xc, yc) : (X, y)
      c = intercept ? λ .* ω : fill(λ, p)
      sol = lasso(X, y, λ; intercept, standardizeX = intercept, lower, upper, optTol = 1e-12)
      b = Vector(sol.x)
      @test inbox(b, l, u)
      @test kkt_box(-Z' * (z - Z * b) / n, b, c, l, u, l1(c)) < 1e-7
      @test sol.penalty isa ProximalBase.BoxConstrained
      intercept && @test sol.intercept ≈ mean(y) - dot(xbar, b)
      # on the Gram matrix, and with the loadings given
      @test Vector(lasso(X, y, λ; intercept, standardizeX = intercept, lower, upper, gram = true,
                         optTol = 1e-12).x) ≈ b atol = 1e-6
      @test Vector(lasso(X, y, λ, c ./ λ; intercept, lower, upper, optTol = 1e-12).x) ≈ b atol = 1e-6
      # solved along a path from λmax when warmStart is false
      @test Vector(lasso(X, y, λ; intercept, standardizeX = intercept, lower, upper, warmStart = false,
                         optTol = 1e-12).x) ≈ b atol = 1e-6
    end
  end

  @testset "no bounds: the lasso as before" begin
    a = lasso(X, y, λ; intercept = true, standardizeX = true, rng = Xoshiro(1))
    for (lower, upper) in ((-Inf, Inf), (fill(-Inf, p), fill(Inf, p)))
      b = lasso(X, y, λ; intercept = true, standardizeX = true, lower, upper, rng = Xoshiro(1))
      @test b.penalty isa NormL1 && Vector(b.x) == Vector(a.x) && b.iterations == a.iterations
    end
    pa = LassoPath(X, y; nλ = 20, rng = Xoshiro(2))
    pb = LassoPath(X, y; nλ = 20, lower = -Inf, rng = Xoshiro(2))
    @test pb.penalty isa NormL1 && pa.λpath == pb.λpath && pa.βpath == pb.βpath
  end

  @testset "the nonnegative lasso: lower = 0 or NormL1Nonneg" begin
    a = lasso(X, y, λ; intercept = true, standardizeX = true, lower = 0, optTol = 1e-12)
    b = lasso(X, y, λ; intercept = true, standardizeX = true, penalty = NormL1Nonneg(1), optTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-7
  end

  @testset "LassoPath" begin
    l, u = zeros(p), fill(Inf, p)
    path = LassoPath(X, y; intercept = true, lower = 0, nλ = 30, optTol = 1e-12)
    # λmax: the largest gradient pointing into the box
    @test path.λpath[1] ≈ maximum(max.(Xc' * yc / n, 0) ./ ω) * (1 + sqrt(eps()))
    @test path.λpath[1] < maximum(abs.(Xc' * yc / n) ./ ω)
    @test nnz(path.βpath[1]) == 0 && nnz(path.βpath[2]) > 0
    @test nnz(lasso(X, y, path.λpath[1] * (1 - 1e-3); intercept = true, standardizeX = true, lower = 0).x) == 1
    for k in eachindex(path.λpath)
      b = Vector(path.βpath[k])
      c = path.λpath[k] .* ω
      @test inbox(b, l, u)
      @test kkt_box(-Xc' * (yc - Xc * b) / n, b, c, l, u, l1(c)) < 1e-7
      @test path.β0path[k] ≈ mean(y) - dot(xbar, b)
    end
    # without screening, on the residuals, and each level from zero
    for kw in ((screening = false,), (gram = false,), (warmStart = false,))
      q = LassoPath(X, y, path.λpath; intercept = true, lower = 0, optTol = 1e-12, kw...)
      @test all(k -> isapprox(Vector(q.βpath[k]), Vector(path.βpath[k]); atol = 1e-6), eachindex(path.λpath))
    end
    # per-coefficient bounds
    pv = LassoPath(X, y; intercept = true, lower = lv, upper = uv, nλ = 30, optTol = 1e-12)
    @test pv.penalty.box.l == lv && pv.penalty.box.u == uv
    for k in eachindex(pv.λpath)
      b = Vector(pv.βpath[k])
      c = pv.λpath[k] .* ω
      @test inbox(b, lv, uv)
      @test kkt_box(-Xc' * (yc - Xc * b) / n, b, c, lv, uv, l1(c)) < 1e-7
    end
    @test all(b -> b[2] == 0, pv.βpath)
    # cross-validation passes the bounds to its paths
    cv = CVLassoPath(X, y; intercept = true, lower = 0, nfolds = 5, nλ = 20, rng = Xoshiro(3))
    @test all(>=(0), coef(cv)[2:end])
  end

  @testset "other penalties on the standardized columns" begin
    s = ω
    Xs = Xc ./ transpose(s)
    gs(bt) = -Xs' * (yc - Xs * bt) / n
    l, u = zeros(p), fill(0.7, p)
    # the bounds apply to the coefficients of the data, so to s .* β̃ on the
    # standardized columns
    for (pen, dpen) in ((MCP(1, 3), (λ, a) -> (j, t) -> sign(t) * max(λ - abs(t) / a, 0.0)),
                        (SCAD(1, 3.7), (λ, a) -> (j, t) -> sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))))
      sol = lasso(X, y, λ; intercept = true, standardizeX = true, penalty = pen, lower = 0, upper = 0.7, optTol = 1e-12)
      b = Vector(sol.x)
      @test inbox(b, l, u; tol = 1e-12)
      @test sol.penalty.box.u ≈ 0.7 .* s
      bt = s .* b
      @test kkt_box(gs(bt), bt, fill(λ, p), s .* l, s .* u, dpen(λ, pen.a)) < 1e-6
      path = LassoPath(X, y; intercept = true, penalty = pen, lower = 0, upper = 0.7, nλ = 20)
      @test all(b -> inbox(Vector(b), l, u; tol = 1e-12), path.βpath)
      @test nnz(path.βpath[1]) == 0
    end
    # glmnet's elastic net
    α = 0.5
    sol = lasso(X, y, λ; intercept = true, standardizeX = true, alpha = α, lower = 0, optTol = 1e-12)
    bt = s .* Vector(sol.x)
    @test all(>=(0), bt)
    @test kkt_box(gs(bt), bt, fill(λ * α, p), zeros(p), fill(Inf, p), (j, t) -> λ * α * sign(t) + λ * (1 - α) * t) < 1e-7
    path = LassoPath(X, y; intercept = true, alpha = α, lower = 0, nλ = 20)
    @test path.λpath[1] ≈ maximum(max.(Xs' * yc / n, 0)) / α * (1 + sqrt(eps()))
    # the adaptive lasso: both fits take the bounds
    sol = adaptiveLasso(X, y, λ; intercept = true, standardizeX = true, lower = 0)
    @test all(>=(0), Vector(sol.x)) && sol.penalty isa ProximalBase.BoxConstrained
  end

  @testset "observation weights and a sparse X" begin
    w = rand(Xoshiro(4), 1:3, n)
    rows = reduce(vcat, [fill(i, w[i]) for i in 1:n])
    a = lasso(X, y, λ; intercept = true, standardizeX = true, weights = w, lower = -0.3, upper = 0.7, optTol = 1e-12)
    b = lasso(X[rows, :], y[rows], λ; intercept = true, standardizeX = true, lower = -0.3, upper = 0.7, optTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-7
    Xsp = sparse(X .* (rand(Xoshiro(5), n, p) .< 0.3))
    a = lasso(Xsp, y, λ; intercept = true, standardizeX = true, lower = 0, optTol = 1e-12)
    b = lasso(Matrix(Xsp), y, λ; intercept = true, standardizeX = true, lower = 0, optTol = 1e-12)
    @test Vector(a.x) ≈ Vector(b.x) atol = 1e-7
  end

  @testset "glmLasso and GLMLassoPath" begin
    yl = Float64.(rand(Xoshiro(6), n) .< logistic.(0.3 .+ X * β))
    l, u = fill(-0.2, p), fill(Inf, p)
    tight = CDOptions(CDOptions(); optTol = 1e-12)
    sol = glmLasso(X, yl, 0.02, ProximalBase.Logistic(), tight; lower = -0.2, newtonTol = 1e-12)
    b = Vector(sol.x)
    @test inbox(b, l, u)
    μ = logistic.(sol.intercept .+ X * b)
    c = 0.02 .* ω
    @test kkt_box(X' * (μ - yl) / n, b, c, l, u, l1(c)) < 1e-6
    path = GLMLassoPath(X, yl; lower = 0, nλ = 20)
    @test path.λpath[1] ≈ maximum(max.(Xc' * (yl .- mean(yl)) / n, 0) ./ ω) * (1 + sqrt(eps())) rtol = 1e-8
    @test all(b -> all(>=(0), b), path.βpath)
    @test nnz(path.βpath[1]) == 0 && nnz(path.βpath[end]) > 0
    # along a path of options.numSteps levels from λmax
    sw = glmLasso(X, yl, 0.02, ProximalBase.Logistic(), CDOptions(tight; warmStart = false); lower = -0.2,
                  newtonTol = 1e-12)
    @test Vector(sw.x) ≈ b atol = 1e-5
    @test_throws ArgumentError glmLasso(X, yl, 0.02; upper = -1)
  end

  @testset "errors" begin
    @test_throws ArgumentError lasso(X, y, λ; lower = 0.1)
    @test_throws ArgumentError lasso(X, y, λ; upper = -0.1)
    @test_throws ArgumentError lasso(X, y, λ; lower = NaN)
    @test_throws ArgumentError lasso(X, y, λ; upper = [NaN; ones(p - 1)])
    @test_throws DimensionMismatch lasso(X, y, λ; lower = zeros(p - 1))
    @test_throws DimensionMismatch LassoPath(X, y; upper = ones(p + 1))
    @test_throws ArgumentError LassoPath(X, y; lower = 1)
    @test_throws ArgumentError lasso(X, y, λ; penalty = Constrained(NormL1(1.0), IndBox(0.0, Inf)))
  end
end

end
