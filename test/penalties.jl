module PenaltiesTest

using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# the largest violation of the optimality conditions of ‖yc - Xc b‖²/(2n) + ∑ⱼ pen(bⱼ)
# at b, for a penalty whose derivative is dpen(t) at t ≠ 0 and whose
# subdifferential at zero is [-λ0, λ0]
function kkt(Xc, yc, b, λ0, dpen)
  grad = -Xc' * (yc - Xc * b) / size(Xc, 1)
  maximum(j -> b[j] == 0 ? max(abs(grad[j]) - λ0, 0.0) : abs(grad[j] + dpen(b[j])), eachindex(b))
end
denet(λ, α) = t -> λ * α * sign(t) + λ * (1 - α) * t
dmcp(λ, a) = t -> sign(t) * max(λ - abs(t) / a, 0.0)
dscad(λ, a) = t -> sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))
atlevel(pen, λ) = pen isa MCP ? MCP(λ, pen.a) : SCAD(λ, pen.a)

@testset "penalties in lasso and LassoPath, and adaptiveLasso" begin
  rng = Xoshiro(71)
  n, p = 120, 30
  X = randn(rng, n, p) .* transpose(1 .+ 2 .* rand(rng, p))     # columns on different scales
  β = zeros(p); β[1:5] .= [1.0, -0.8, 0.6, 0.5, -0.4]
  y = 0.5 .+ X * β .+ randn(rng, n)
  # the standardized columns of glmnet and ncvreg
  xbar = vec(mean(X; dims = 1))
  s = vec(sqrt.(mean((X .- transpose(xbar)) .^ 2; dims = 1)))
  Xc = (X .- transpose(xbar)) ./ transpose(s)
  yc = y .- mean(y)

  @testset "NormL1 as the penalty is the lasso" begin
    for kw in ((intercept = false, standardizeX = false), (intercept = true, standardizeX = true))
      a = lasso(X, y, 0.1; kw..., rng = Xoshiro(1))
      b = lasso(X, y, 0.1; kw..., penalty = NormL1(7.0), rng = Xoshiro(1))
      @test Vector(a.x) == Vector(b.x) && a.iterations == b.iterations
      @test b.penalty.λ0 == 0.1 && b.penalty.w == a.penalty.w && b.scales === nothing
      pa = LassoPath(X, y; nλ = 20, kw..., rng = Xoshiro(2))
      pb = LassoPath(X, y; nλ = 20, kw..., penalty = NormL1(1), rng = Xoshiro(2))
      @test pa.λpath == pb.λpath && pa.βpath == pb.βpath && pa.iterations == pb.iterations
      @test pb.penalty.w == pa.penalty.w && pa.penalty isa NormL1 && pa.scales === nothing
      @test Vector(lasso(X, y, 0.1; kw..., alpha = 1, rng = Xoshiro(1)).x) == Vector(a.x)
    end
    # the weights of the template multiply the loadings
    v = rand(Xoshiro(3), p) .+ 0.5
    a = lasso(X, y, 0.1, v .* s; intercept = true, optTol = 1e-12)
    b = lasso(X, y, 0.1; intercept = true, standardizeX = true, penalty = NormL1(1, v), optTol = 1e-12)
    @test b.penalty.w ≈ v .* s
    @test Vector(b.x) ≈ Vector(a.x) atol = 1e-8
    # without standardization, the weights of the template are the loadings
    a = lasso(X, y, 0.1, v; rng = Xoshiro(1))
    b = lasso(X, y, 0.1; penalty = NormL1(1, v), rng = Xoshiro(1))
    @test Vector(a.x) == Vector(b.x) && b.penalty.w == v
    # the nonnegative lasso, also along cold-started paths
    c = lasso(X, y, 0.02; penalty = NormL1Nonneg(1), intercept = true)
    @test all(>=(0), Vector(c.x)) && count(!iszero, Vector(c.x)) > 0
    pn = LassoPath(X, y; penalty = NormL1Nonneg(1), nλ = 8, intercept = true, optTol = 1e-12)
    pc = LassoPath(X, y, pn.λpath; penalty = NormL1Nonneg(1), intercept = true, optTol = 1e-12, warmStart = false)
    @test all(b -> all(>=(0), b), pn.βpath) && pn.penalty isa NormL1Nonneg
    @test all(isapprox(Vector(pc.βpath[k]), Vector(pn.βpath[k]); atol = 1e-8) for k in 1:8)
  end

  @testset "the elastic net of glmnet" begin
    for α in (0.2, 0.5, 0.8)
      λ = 0.05
      sol = lasso(X, y, λ; alpha = α, intercept = true, standardizeX = true, optTol = 1e-12)
      @test sol.penalty isa Scaled && sol.penalty.α == λ && sol.penalty.f.λ1 == α && sol.penalty.f.λ2 == 1 - α
      @test sol.scales ≈ s
      b̃ = Vector(sol.x) .* s
      @test kkt(Xc, yc, b̃, λ * α, denet(λ, α)) < 1e-8
      @test sol.intercept ≈ mean(y) - dot(xbar, Vector(sol.x))
      @test predict(sol, X) ≈ y - sol.residuals
      # ElasticNet on the standardized columns, solved directly
      x = SparseIterate(p)
      solve!(x, (1 / n) * LeastSquaresLoss(yc, Xc), ElasticNet(λ * α, λ * (1 - α)), CDSolver(; tol = 1e-12))
      @test b̃ ≈ Vector(x) atol = 1e-8
    end
    # the path: glmnet's λmax for α, a zero first solution, the optimality conditions at every level
    α = 0.5
    path = LassoPath(X, y; alpha = α, nλ = 25, intercept = true, optTol = 1e-12)
    @test path.λpath[1] ≈ maximum(abs.(Xc' * yc / n)) / α * (1 + sqrt(eps()))
    @test nnz(path.βpath[1]) == 0 && nnz(path.βpath[end]) > 5
    @test path.scales ≈ s && path.penalty isa Scaled && path.penalty.α == 1
    for k in eachindex(path.λpath)
      b̃ = Vector(path.βpath[k]) .* s
      @test kkt(Xc, yc, b̃, path.λpath[k] * α, denet(path.λpath[k], α)) < 1e-8
      @test path.β0path[k] ≈ mean(y) - dot(xbar, Vector(path.βpath[k]))
    end
    # the same with the Gram matrix, without screening, from lasso, and with
    # warmStart = false
    for kw in ((gram = false,), (gram = true,), (screening = false,), (warmStart = false,))
      pk = LassoPath(X, y, path.λpath[1:12]; alpha = α, intercept = true, optTol = 1e-12, kw...)
      @test all(isapprox(Vector(pk.βpath[k]), Vector(path.βpath[k]); atol = 1e-8) for k in 1:12)
      @test pk.β0path ≈ path.β0path[1:12]
    end
    s10 = lasso(X, y, path.λpath[10]; alpha = α, intercept = true, standardizeX = true, optTol = 1e-12)
    @test Vector(s10.x) ≈ Vector(path.βpath[10]) atol = 1e-8
    @test coef(path, path.λpath[10]) ≈ coef(s10) atol = 1e-8
    # without standardization the penalty applies to the coefficients themselves
    sol = lasso(X, y, 0.05; alpha = 0.5, intercept = true, optTol = 1e-12)
    @test sol.scales === nothing
    @test kkt(X .- transpose(xbar), yc, Vector(sol.x), 0.025, denet(0.05, 0.5)) < 1e-8
    # the scaled elastic net as a template is alpha
    st = lasso(X, y, 0.05; penalty = Scaled(ElasticNet(0.5, 0.5), 3.0), intercept = true, standardizeX = true, optTol = 1e-12)
    sa = lasso(X, y, 0.05; alpha = 0.5, intercept = true, standardizeX = true, optTol = 1e-12)
    @test st.penalty.α == 0.05
    @test Vector(st.x) ≈ Vector(sa.x) atol = 1e-10
    # ElasticNet(1, λ2) keeps λ2 fixed along the path
    pe = LassoPath(X, y; penalty = ElasticNet(1, 0.3), nλ = 10, intercept = true, optTol = 1e-12)
    for k in eachindex(pe.λpath)
      b̃ = Vector(pe.βpath[k]) .* s
      @test kkt(Xc, yc, b̃, pe.λpath[k], t -> pe.λpath[k] * sign(t) + 0.3t) < 1e-8
    end
  end

  @testset "MCP and SCAD: ncvreg's standardization and path" begin
    for (pen, d) in ((MCP(1, 3), dmcp), (SCAD(1, 3.7), dscad))
      path = LassoPath(X, y; penalty = pen, nλ = 30, intercept = true, optTol = 1e-12)
      @test path.λpath[1] ≈ maximum(abs.(Xc' * yc / n)) * (1 + sqrt(eps()))
      @test nnz(path.βpath[1]) == 0 && path.scales ≈ s
      for k in eachindex(path.λpath)
        b̃ = Vector(path.βpath[k]) .* s
        @test kkt(Xc, yc, b̃, path.λpath[k], d(path.λpath[k], pen.a)) < 1e-8
      end
      # lasso follows a path from λmax to its level
      λ = path.λpath[20]
      sol = lasso(X, y, λ; penalty = pen, intercept = true, standardizeX = true, optTol = 1e-12)
      @test kkt(Xc, yc, Vector(sol.x) .* s, λ, d(λ, pen.a)) < 1e-8
      @test sol.penalty == atlevel(pen, λ)
      sol0 = lasso(X, y, λ; penalty = pen, intercept = true, standardizeX = true, optTol = 1e-12, numSteps = 0)
      @test kkt(Xc, yc, Vector(sol0.x) .* s, λ, d(λ, pen.a)) < 1e-8
    end
    # where the objective is convex (a > 1/the smallest eigenvalue of Xc'Xc/n), the
    # solution is unique and equals that of the penalty solved directly
    @test 1 / eigmin(Symmetric(Xc' * Xc / n)) < 6
    for pen in (MCP(1, 6), SCAD(1, 6))
      path = LassoPath(X, y; penalty = pen, nλ = 15, intercept = true, optTol = 1e-12)
      for k in (5, 10, 15)
        x = SparseIterate(p)
        solve!(x, (1 / n) * LeastSquaresLoss(yc, Xc), atlevel(pen, path.λpath[k]), CDSolver(; tol = 1e-12))
        @test Vector(path.βpath[k]) .* s ≈ Vector(x) atol = 1e-7
      end
    end
    # the large coefficients of MCP are less biased than those of the lasso: at
    # the first level where the lasso selects the five, MCP estimates them better
    pl = LassoPath(X, y; nλ = 30, intercept = true)
    pm = LassoPath(X, y, pl.λpath; penalty = MCP(1, 3), intercept = true)
    k = findfirst(b -> all(!iszero, b[1:5]), pl.βpath)
    @test norm(Vector(pm.βpath[k])[1:5] - β[1:5]) < norm(Vector(pl.βpath[k])[1:5] - β[1:5])
  end

  @testset "local linear approximation (lla)" begin
    λ = 0.1
    # the two-step estimator of Fan, Xue and Zou: the lasso, then the lasso with
    # the weights p′(|b|)/λ at its solution, on the standardized columns
    wmcp(b, λ, a) = max.(1 .- abs.(b) ./ (a * λ), 0.0)
    wscad(b, λ, a) = [abs(t) <= λ ? 1.0 : max(a * λ - abs(t), 0.0) / ((a - 1) * λ) for t in b]
    b1 = Vector(lasso(Xc, yc, λ; optTol = 1e-12).x)
    for (pen, w) in ((MCP(1, 3), wmcp(b1, λ, 3)), (SCAD(1, 3.7), wscad(b1, λ, 3.7)))
      sol = lasso(X, y, λ; penalty = pen, lla = LLA(), intercept = true, standardizeX = true, optTol = 1e-12)
      b2 = lasso(Xc, yc, λ, w; optTol = 1e-12).x
      @test Vector(sol.x) .* s ≈ Vector(b2) atol = 1e-7
      @test sol.converged && sol.penalty == atlevel(pen, λ)
    end

    # CappedL1 by default iterates until the weights stop changing: the solution
    # satisfies the optimality conditions of the weighted lasso with the weights
    # (|bⱼ| < a) at itself, and the entries beyond the cap are not shrunk
    a = 0.5
    sol = lasso(X, y, λ; penalty = CappedL1(1, a), intercept = true, standardizeX = true, optTol = 1e-12)
    b̃ = Vector(sol.x) .* s
    w = Float64.(abs.(b̃) .< a)
    grad = -Xc' * (yc - Xc * b̃) / n
    @test maximum(j -> b̃[j] == 0 ? max(abs(grad[j]) - λ * w[j], 0.0) : abs(grad[j] + λ * w[j] * sign(b̃[j])), 1:p) < 1e-8
    @test count(abs.(b̃) .> a) >= 3 && sol.converged && sol.penalty == CappedL1(λ, a)

    # its path starts at λmax of the lasso, the slope at zero being λ, and with
    # start = :lasso each level is that of lasso at the level
    path = LassoPath(X, y; penalty = CappedL1(1, a), nλ = 20, intercept = true, optTol = 1e-12)
    @test path.λpath[1] ≈ maximum(abs.(Xc' * yc / n)) * (1 + sqrt(eps()))
    @test nnz(path.βpath[1]) == 0 && all(path.converged)
    for k in (5, 10, 20)
      solk = lasso(X, y, path.λpath[k]; penalty = CappedL1(1, a), intercept = true, standardizeX = true, optTol = 1e-12)
      @test Vector(path.βpath[k]) ≈ Vector(solk.x) atol = 1e-6
    end
    # with warmStart = false each level is solved from zero, to the same solutions
    pc = LassoPath(X, y, path.λpath[1:5]; penalty = CappedL1(1, a), intercept = true, warmStart = false, optTol = 1e-12)
    @test all(k -> isapprox(Vector(pc.βpath[k]), Vector(path.βpath[k]); atol = 1e-6), 1:5)

    # LogPenalty(λ, ε) has the slope λ/ε at zero, so λmax is ε times that of the lasso
    pathl = LassoPath(X, y; penalty = LogPenalty(1, 0.5), nλ = 10, intercept = true)
    @test pathl.λpath[1] ≈ 0.5 * maximum(abs.(Xc' * yc / n)) * (1 + sqrt(eps()))
    @test nnz(pathl.βpath[1]) == 0 && nnz(pathl.βpath[end]) > 0

    # start = :previous: lasso follows numSteps levels from λmax, as the path on
    # the same levels does; with numSteps = 0 it starts from the lasso at λ
    ll = LLA(; stages = 100, start = :previous)
    λ10 = path.λpath[10]
    sol = lasso(X, y, λ10; penalty = MCP(1, 3), lla = ll, intercept = true, standardizeX = true, optTol = 1e-12,
                numSteps = 8)
    λs = exp.(range(log(maximum(abs.(Xc' * yc / n))), log(λ10); length = 9))
    pp = LassoPath(X, y, λs; penalty = MCP(1, 3), lla = ll, intercept = true, optTol = 1e-12)
    @test Vector(sol.x) ≈ Vector(pp.βpath[end]) atol = 1e-8
    sol0 = lasso(X, y, λ10; penalty = MCP(1, 3), lla = ll, intercept = true, standardizeX = true, optTol = 1e-12,
                 numSteps = 0)
    sol1 = lasso(X, y, λ10; penalty = MCP(1, 3), lla = LLA(; stages = 100), intercept = true, standardizeX = true,
                 optTol = 1e-12)
    @test Vector(sol0.x) ≈ Vector(sol1.x) atol = 1e-10

    # the exact steps of MCP and SCAD refuse a column of variance at most 1/a
    # without standardizeX; standardizeX or lla avoid the condition
    Xs = copy(X); Xs[:, 4] .*= 0.1 / s[4]                 # variance 0.01
    @test_throws ArgumentError lasso(Xs, y, λ; penalty = MCP(1, 3), intercept = true)
    @test_throws ArgumentError LassoPath(Xs, y; penalty = SCAD(1, 3.7), standardizeX = false, intercept = true)
    @test lasso(Xs, y, λ; penalty = MCP(1, 3), intercept = true, standardizeX = true).converged
    soll = lasso(Xs, y, λ; penalty = MCP(1, 3), lla = LLA(), intercept = true, optTol = 1e-12)
    c1 = Vector(lasso(Xs, y, λ; intercept = true, optTol = 1e-12).x)
    @test Vector(soll.x) ≈ Vector(lasso(Xs, y, λ, wmcp(c1, λ, 3); intercept = true, optTol = 1e-12).x) atol = 1e-7

    # CVLassoPath gives lla to its paths
    cv = CVLassoPath(X, y; penalty = MCP(1, 3), lla = LLA(), nλ = 15, intercept = true, rng = Xoshiro(1))
    full = LassoPath(X, y; penalty = MCP(1, 3), lla = LLA(), nλ = 15, intercept = true)
    @test cv.λpath ≈ full.λpath
    @test Vector(cv.path.βpath[end]) ≈ Vector(full.βpath[end]) atol = 1e-6

    # lla is for these four penalties, without bounds, and the GLM fits refuse
    # CappedL1 and LogPenalty
    @test_throws ArgumentError lasso(X, y, λ; lla = LLA())
    @test_throws ArgumentError lasso(X, y, λ; alpha = 0.5, lla = LLA())
    @test_throws ArgumentError lasso(X, y, λ; penalty = ElasticNet(1, 0.1), lla = LLA())
    @test_throws ArgumentError LassoPath(X, y; lla = LLA())
    @test_throws ArgumentError lasso(X, y, λ; penalty = CappedL1(1, a), lower = 0)
    @test_throws ArgumentError lasso(X, y, λ; penalty = MCP(1, 3), lla = LLA(), upper = 1)
    @test_throws ArgumentError LassoPath(X, y; penalty = LogPenalty(1, 1), lower = -1)
    yb = Float64.(y .> median(y))
    @test_throws ArgumentError glmLasso(X, yb, 0.01; penalty = CappedL1(1, a))
    @test_throws ArgumentError GLMLassoPath(X, yb; penalty = LogPenalty(1, 1))
  end

  @testset "observation weights, sparse X, Float32" begin
    w = rand(Xoshiro(8), 1:3, n)
    idx = reduce(vcat, [fill(i, w[i]) for i in 1:n])
    for pen in (nothing, MCP(1, 6))
      kw = pen === nothing ? (alpha = 0.5,) : (penalty = pen,)
      a = lasso(X, y, 0.05; kw..., intercept = true, standardizeX = true, weights = w, optTol = 1e-12)
      b = lasso(X[idx, :], y[idx], 0.05; kw..., intercept = true, standardizeX = true, optTol = 1e-12)
      @test Vector(a.x) ≈ Vector(b.x) atol = 1e-7
      @test a.intercept ≈ b.intercept atol = 1e-7
    end
    Xsp = sparse(X .* (rand(Xoshiro(9), n, p) .< 0.3))
    ps = LassoPath(Xsp, y; alpha = 0.5, nλ = 10, intercept = true, optTol = 1e-12)
    pd = LassoPath(Matrix(Xsp), y, ps.λpath; alpha = 0.5, intercept = true, optTol = 1e-12)
    @test all(isapprox(Vector(ps.βpath[k]), Vector(pd.βpath[k]); atol = 1e-7) for k in eachindex(ps.λpath))
    s32 = lasso(Float32.(X), Float32.(y), 0.05; alpha = 0.5, intercept = true, standardizeX = true)
    @test eltype(s32.x) == Float32 && s32.penalty.f isa ElasticNet{Float32}
    @test LassoPath(Float32.(X), Float32.(y); penalty = SCAD(1, 3.7), nλ = 5).λpath isa Vector{Float32}
  end

  @testset "adaptiveLasso" begin
    λ = 0.05
    # example 01: loadings from least squares
    β̃ = X \ y
    a = adaptiveLasso(X, y, λ; init = β̃, rng = Xoshiro(5))
    b = lasso(X, y, λ, 1 ./ abs.(β̃); rng = Xoshiro(5))
    @test Vector(a.x) == Vector(b.x) && a.penalty.w ≈ 1 ./ abs.(β̃)
    # from a lasso: the two steps by hand, with the same generator; the zeros of
    # the first stay zero
    a = adaptiveLasso(X, y, λ; intercept = true, standardizeX = true, initλ = 0.1, γ = 2, rng = Xoshiro(6))
    r = Xoshiro(6)
    first = lasso(X, y, 0.1; intercept = true, standardizeX = true, rng = r)
    ω = [iszero(v) ? Inf : 1 / abs(v)^2 for v in Vector(first.x)]
    second = lasso(X, y, λ, ω; intercept = true, rng = r)
    @test a.penalty.w ≈ ω
    @test Vector(a.x) ≈ Vector(second.x) atol = 1e-10
    @test a.intercept ≈ second.intercept atol = 1e-10
    @test all(iszero, Vector(a.x)[iszero.(Vector(first.x))])
    @test issubset(1:5, findall(!iszero, Vector(a.x)))
    # ridge regression as the first estimate
    Xm = X .- transpose(xbar)
    rd = lasso(X, y, 0.0; penalty = ElasticNet(0, 0.5), intercept = true, optTol = 1e-12)
    @test Vector(rd.x) ≈ (Xm' * Xm / n + 0.5I) \ (Xm' * yc / n) atol = 1e-8
    a = adaptiveLasso(X, y, λ; init = :ridge, initλ = 0.5, intercept = true, optTol = 1e-12)
    @test a.penalty.w ≈ 1 ./ abs.(Vector(rd.x)) rtol = 1e-6
    @test adaptiveLasso(round.(Int, X), round.(Int, 10y), 1).x isa SparseIterate{Float64}
    @test_throws ArgumentError adaptiveLasso(X, y, λ; init = :bogus)
    @test_throws ArgumentError adaptiveLasso(X, y, 0.0)
    @test_throws ArgumentError adaptiveLasso(X, y, λ; γ = 0)
    @test_throws DimensionMismatch adaptiveLasso(X, y, λ; init = ones(3))
  end

  @testset "errors" begin
    @test_throws ArgumentError lasso(X, y, 0.1; alpha = 0)
    @test_throws ArgumentError lasso(X, y, 0.1; alpha = 1.5)
    @test_throws ArgumentError lasso(X, y, 0.1; alpha = 0.5, penalty = MCP(1, 3))
    @test_throws ArgumentError lasso(X, y, 0.1; penalty = NormL2(1.0))
    @test_throws ArgumentError LassoPath(X, y; penalty = IndBox(-1, 1))
  end
end

end
