module HDMTest

using HighDimensionalModels, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test
using SpecialFunctions: erfcinv

qnorm(t) = -sqrt(2) * erfcinv(2t)

@testset "rlasso, post_lasso and double_selection" begin
  @testset "theory_lambda" begin
    n, p = 100, 200
    @test theory_lambda(n, p) ≈ 1.1 * qnorm(1 - 0.1 / log(n) / (2p)) / sqrt(n)
    @test theory_lambda(n, p; c = 0.5, γ = 0.05) ≈ 0.5 * qnorm(1 - 0.05 / (2p)) / sqrt(n)
    @test theory_lambda(n, 2p) > theory_lambda(n, p) > theory_lambda(4n, p)
    @test_throws ArgumentError theory_lambda(n, p; γ = 1.5)
  end

  rng = Xoshiro(51)
  n, p = 300, 100
  X = randn(rng, n, p)
  β = zeros(p); β[1:4] .= [1.0, -0.8, 0.6, 0.5]
  y = 1.0 .+ X * β .+ (0.5 .+ abs.(X[:, 5])) .* randn(rng, n)    # heteroskedastic

  @testset "rlasso" begin
    sol = rlasso(X, y)
    @test issubset(1:4, sol.selected)
    @test sol.post && sol.iterations >= 1
    # the coefficients are least squares on the selected columns, with an intercept
    Z = [ones(n) X[:, sol.selected]]
    b = Z \ y
    @test sol.intercept ≈ b[1] atol = 1e-10
    @test Vector(sol.x)[sol.selected] ≈ b[2:end] atol = 1e-10
    @test sol.residuals ≈ y .- Z * b atol = 1e-10
    @test sol.σ ≈ std(sol.residuals)
    @test sol.λ0 ≈ theory_lambda(n, p)
    # the loadings are those of the residuals
    Xc = X .- mean(X; dims = 1)
    @test sol.loadings ≈ vec(sqrt.(mean((Xc .* sol.residuals) .^ 2; dims = 1)))
    # without the refit, the lasso coefficients and hdm's c = 0.5
    sol0 = rlasso(X, y; post = false)
    @test sol0.λ0 ≈ theory_lambda(n, p; c = 0.5) && !sol0.post
    @test issubset(1:4, sol0.selected)
    # pure noise: nothing selected, the intercept is the mean
    soln = rlasso(X, randn(rng, n) .+ 2)
    @test isempty(soln.selected) || length(soln.selected) <= 2
    # integer data
    @test rlasso(round.(Int, 10X), round.(Int, 10y)).selected isa Vector{Int}
  end

  @testset "post_lasso" begin
    sol = lasso(X, y, 0.1; intercept = true)
    pl = post_lasso(X, y, sol)
    S = pl.selected
    @test S == findall(!iszero, Vector(sol.x))
    Z = [ones(n) X[:, S]]
    b = Z \ y
    e = y .- Z * b
    @test pl.intercept ≈ b[1] && Vector(pl.x)[S] ≈ b[2:end]
    # HC1 by its definition
    A = inv(Z' * Z)
    V = n / (n - size(Z, 2)) * A * (Z' * Diagonal(e .^ 2) * Z) * A
    @test pl.vcov ≈ V rtol = 1e-8
    @test pl.se_intercept ≈ sqrt(V[1, 1]) && Vector(pl.se)[S] ≈ sqrt.(diag(V)[2:end])
    # from rlasso, or from a vector of columns, and without an intercept
    @test post_lasso(X, y, rlasso(X, y)).x ≈ rlasso(X, y).x
    pl0 = post_lasso(X, y, [1, 2, 3]; intercept = false)
    @test pl0.intercept == 0 && Vector(pl0.x)[1:3] ≈ X[:, 1:3] \ y
    @test_throws ArgumentError post_lasso(X[1:3, :], y[1:3], [1, 2, 3])
    @test post_lasso(round.(Int, 10X), round.(Int, 10y), [1, 2]).selected == [1, 2]
  end

  @testset "double_selection" begin
    # the treatment depends on the controls that also enter y: selecting on y
    # alone would drop them and bias the estimate
    α = 0.5
    rng2 = Xoshiro(52)
    m, q = 400, 150
    W = randn(rng2, m, q)
    d = W[:, 1:3] * [1.0, 1.0, -1.0] .+ randn(rng2, m)
    yd = α .* d .+ W[:, 1:3] * [1.0, 1.0, -1.0] .+ randn(rng2, m)
    r = double_selection(yd, d, W)
    @test r.method === :doubleselection && r.n == m
    @test issubset(1:3, r.selected)
    @test abs(r.estimate - α) < 4 * r.se
    @test r.ci[1] < r.estimate < r.ci[2]
    @test r.ci[2] - r.ci[1] ≈ 2 * qnorm(0.975) * r.se
    @test r.t ≈ r.estimate / r.se && 0 <= r.pvalue <= 1
    # the estimate is least squares of y on d and the selected controls
    Z = [ones(m) d W[:, r.selected]]
    @test r.estimate ≈ (Z \ yd)[2] atol = 1e-10
    # partialling out gives nearly the same estimate here
    rp = double_selection(yd, d, W; method = :partialling)
    @test rp.method === :partialling && abs(rp.estimate - r.estimate) < 0.5 * r.se
    # forced controls enter the regression
    rf = double_selection(yd, d, W; forced = [100])
    @test 100 in rf.selected
    # a wider interval at a higher level
    r99 = double_selection(yd, d, W; level = 0.99)
    @test r99.ci[2] - r99.ci[1] > r.ci[2] - r.ci[1]
    @test_throws ArgumentError double_selection(yd, d, W; method = :bogus)
    @test double_selection(round.(Int, 10yd), round.(Int, 10d), round.(Int, 10W)).n == m
    @test occursin("TreatmentEffect", sprint(show, r))
  end

  @testset "feasible_lasso! at the theory level" begin
    x = SparseIterate(p)
    sol = feasible_lasso!(x, X, y; intercept = true)
    x2 = SparseIterate(p)
    sol2 = feasible_lasso!(x2, X, y, theory_lambda(n, p); intercept = true)
    @test Vector(sol.x) ≈ Vector(sol2.x) atol = 1e-6
    @test sol.penalty.λ0 ≈ theory_lambda(n, p)
    @test_throws ArgumentError feasible_lasso!(SparseIterate(p), X, y; λ0 = :bogus)
  end
end

@testset "rlasso nuisance record and RNG" begin
  data_rng = Xoshiro(118201)
  n, p = 180, 18
  X = randn(data_rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.35 .* X[:, j-1] .+ sqrt(1 - 0.35^2) .* X[:, j]
  end
  y = 0.7 .+ X[:, 1:3] * [1.7, -1.2, 0.8] .+
      (0.4 .+ 0.2 .* abs.(X[:, 5])) .* randn(data_rng, n)
  options = CDOptions(opt_tol = 1e-10)
  Xc, yc = X .- mean(X; dims = 1), y .- mean(y)

  # These well-scaled, moderately correlated columns have response RMS < 3.
  # A 1e-8 absolute score allowance covers accumulated coordinate errors from
  # the relative 1e-10 step rule over 18 columns, plus floating-point products.
  # No allowance is made for using updated loadings or refitted coefficients.
  kkt_atol = 1e-8
  for (post, tol) in ((true, 1e-5), (false, 1e-5), (true, Inf))
    r = rlasso(X, y, options; post, tol, rng = Xoshiro(118211))
    S = findall(!iszero, r.lasso)
    off = setdiff(1:p, S)
    residual = yc - Xc * r.lasso
    g = Xc' * residual / n
    penalty = r.lasso_λ0 .* r.lasso_loadings
    on_error = maximum(abs, g[S] - penalty[S] .* sign.(r.lasso[S]))
    off_excess = max(0.0, maximum(abs.(g[off]) - penalty[off]))
    @test on_error <= kkt_atol
    @test off_excess <= kkt_atol
    @test r.selected == S
    @test r.rank == length(S)
    @test r.converged && r.solver_converged
    @test r.lasso_loadings !== r.loadings
    @test r.lasso !== r.x && nonzeros(r.lasso) !== nonzeros(r.x)
    @test abs(mean(y .- r.lasso_intercept .- X * r.lasso)) <= 1e-12
    @test r.lasso_intercept ≈ mean(y) - dot(vec(mean(X; dims = 1)), r.lasso) atol = 1e-12 rtol = 0
    @test r.tuning == (; c = post ? 1.1 : 0.5, γ = 0.1 / log(n), max_iter = 15, tol, options)
    if isinf(tol)
      @test r.iterations == 1
      @test r.lasso_λ0 == r.λ0 / 2
    else
      @test r.iterations > 1
      @test r.lasso_λ0 == r.λ0
    end
    if post
      b = qr([ones(n) X[:, S]], ColumnNorm()) \ y
      @test [r.intercept; Vector(r.x)[S]] ≈ b atol = 1e-12 rtol = 0
      @test norm(r.x - r.lasso) > 1e-3
    else
      @test r.x == r.lasso
    end
    # Every field, including the record, must be reproducible with private RNGs.
    a = rlasso(X, y, options; post, tol, rng = Xoshiro(118211))
    for field in fieldnames(typeof(r))
      @test getfield(r, field) == getfield(a, field)
    end
    println("rlasso KKT post=$post tol=$tol on_support=$on_error off_support_excess=$off_excess")
  end

  @testset "termination and effective tuning" begin
    r = rlasso(X, y, options; max_iter = 1, tol = 0, rng = Xoshiro(118212))
    @test !r.converged && r.solver_converged
    @test r.iterations == 1 && r.lasso_λ0 == r.λ0 / 2
    @test r.selected == findall(!iszero, r.lasso)
    @test r.tuning.max_iter == 1 && r.tuning.options.max_iter == options.max_iter

    # One pass cannot meet the stopping rule on this nonzero problem.
    r = @test_logs (:warn, r"coordinate descent did not converge") rlasso(
      X, y, CDOptions(max_iter = 1); tol = Inf, rng = Xoshiro(118213))
    @test r.converged && !r.solver_converged
    @test r.selected == findall(!iszero, r.lasso)

    r = rlasso(X, y, options; c = 1.7, γ = 0.05, max_iter = 3, tol = Inf, rng = Xoshiro(118214))
    @test r.tuning == (; c = 1.7, γ = 0.05, max_iter = 3, tol = Inf, options)
    @test r.λ0 == theory_lambda(n, p; c = 1.7, γ = 0.05)

    # A large penalty makes the first solve empty without changing the data.
    for post in (true, false)
      r = rlasso(X, y, options; post, c = 100.0, rng = Xoshiro(118215))
      @test isempty(r.selected) && r.selected == findall(!iszero, r.lasso)
      @test iszero(r.lasso) && r.lasso_intercept == mean(y)
      @test r.converged && r.solver_converged && r.rank == 0 && r.iterations == 1
      @test r.lasso_λ0 == (post ? r.λ0 / 2 : r.λ0)
      @test r.lasso_loadings == r.loadings && r.lasso_loadings !== r.loadings
      @test r.lasso !== r.x
    end
    # No passes: the empty-return path must retain the solver failure too.
    r = @test_logs (:warn, r"coordinate descent did not converge") rlasso(
      X, y, CDOptions(max_iter = 0); rng = Xoshiro(118216))
    @test isempty(r.selected) && r.converged && !r.solver_converged && r.rank == 0
  end

  @testset "OLS backslash numerics" begin
    # Consolidating the refit/rank helper must retain the old backslash
    # coefficients and residuals exactly, including numerical rank deficiency.
    W = [X[:, 1] 1e-18 .* X[:, 5]]
    z = X[:, 1] .+ X[:, 5]
    for (A, target, S) in ((X, y, [1, 2, 3]), (X, y, Int[]), (W, z, [1, 2]))
      for intercept in (true, false)
        xbar = intercept ? vec(mean(A; dims = 1)) : zeros(size(A, 2))
        ybar = intercept ? mean(target) : 0.0
        Z = A[:, S] .- xbar[S]'
        old_b = isempty(S) ? Float64[] : qr(Z, ColumnNorm()) \ (target .- ybar)
        b, e, _ = HighDimensionalModels._ols(A, target, S, xbar, ybar)
        @test b == old_b
        @test e == (target .- ybar) .- Z * old_b
      end
    end
  end

  @testset "rank and no intercept" begin
    for post in (true, false)
      r = rlasso(X, y, options; post, intercept = false, rng = Xoshiro(118217))
      @test r.intercept == r.lasso_intercept == 0
      @test r.rank == length(r.selected)
      if post
        @test Vector(r.x)[r.selected] ≈ qr(X[:, r.selected], ColumnNorm()) \ y atol = 1e-12 rtol = 0
      end
      # Both directions carry signal for lasso; the tiny column lies below
      # the pivoted-QR solve's numerical-rank threshold in its original units.
      u, v = X[:, 1], X[:, 5]
      W, z = [u 1e-18 .* v], u .+ v
      r = rlasso(W, z, options; post, tol = Inf, rng = Xoshiro(118218))
      @test r.selected == [1, 2]
      @test r.rank == 1
    end
  end

  @testset "default stream, private stream, and real-data wrapper" begin
    state = copy(Random.default_rng())
    rlasso(X, y; rng = Xoshiro(118219))
    @test copy(Random.default_rng()) == state
    Random.seed!(118220)
    a = rlasso(X, y)
    Random.seed!(118220)
    b = rlasso(X, y; rng = Random.default_rng())
    for field in fieldnames(typeof(a))
      @test getfield(a, field) == getfield(b, field)
    end
    Xi, yi = round.(Int, 10X), round.(Int, 10y)
    a = rlasso(Xi, yi; rng = Xoshiro(118221))
    b = rlasso(Float64.(Xi), Float64.(yi); rng = Xoshiro(118221))
    for field in fieldnames(typeof(a))
      @test getfield(a, field) == getfield(b, field)
    end
  end
end

end
