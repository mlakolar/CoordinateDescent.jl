module HDMEffectTest

using HighDimensionalModels, LinearAlgebra, Random, SparseArrays, Statistics, Test
using SpecialFunctions: erfc, erfcinv

const HDM = HighDimensionalModels
const VCOVS = (:hdm, :homoskedastic, :HC0, :HC1, :HC2, :HC3)
const METHODS = (:doubleselection, :partialling)

# Consistent synthetic records isolate the final stage from support selection.
# They deliberately make no claim about the provenance of the penalized fit.
function record(X::Matrix{T}, z::Vector{T}, S = Int[]; post = true) where {T}
  n, p = size(X)
  b = [ones(T, n) X[:, S]] \ z
  x = sparsevec(S, b[2:end], p)
  e = z .- b[1] .- X * x
  HDM.RLassoSolution{T}(x, b[1], copy(S), T(0.1), ones(T, p), e, std(e), 1, post, true,
    copy(x), b[1], T(0.1), ones(T, p), true, true, length(S),
    (; c = post ? 1.1 : 0.5, γ = 0.1 / log(n), max_iter = 15, tol = 1e-5, options = CDOptions()))
end

function changed(r; kwargs...)
  typeof(r)((get(kwargs, f, getfield(r, f)) for f in fieldnames(typeof(r)))...)
end

function same(a, b)
  if a isa Union{HDM.RLassoSolution, HDM.TreatmentEffect}
    return typeof(a) === typeof(b) && all(f -> same(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))
  end
  a == b
end

function rejects(f, quantity; exception = ArgumentError)
  error = try
    f()
    nothing
  catch e
    e
  end
  @test error isa exception
  if error isa Exception
    @test occursin(quantity, sprint(showerror, error))
  end
end

function design(seed = 118301; n = 80, p = 8)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  d = 0.2 .+ X[:, [1, 3]] * [1.4, -1.1] .+ randn(rng, n)
  y = 0.4 .+ 0.7 .* d .+ X[:, [3, 5]] * [1.3, -0.9] .+
      (0.4 .+ 0.3 .* abs.(X[:, 2])) .* randn(rng, n)
  X, y, d
end

# Independent dense regression, with an explicit hat matrix and sandwich.
function dense(y, d, X, fy, fd, method, forced, type)
  n = length(y)
  S = sort(union(fy.selected, fd.selected, method == :doubleselection ? forced : Int[]))
  if method == :doubleselection
    W = [ones(n) X[:, S]]
    Z, response = [ones(n) d X[:, S]], y
    P = W * inv(W' * W) * W'
    v, w, s = d - P * d, y - P * y, length(S)
  else
    Z, response = [ones(n) fd.residuals], fy.residuals
    P = fill(1 / n, n, n)
    v, w, s = fd.residuals .- mean(fd.residuals), fy.residuals .- mean(fy.residuals), 0
  end
  b = Z \ response
  e = response - Z * b
  A = inv(Z' * Z)
  h = diag(Z * A * Z')
  k = size(Z, 2)
  V = if type === :homoskedastic || (type === :hdm && method === :partialling)
    sum(abs2, e) / (n - k) * A[2, 2]
  else
    q = type === :HC2 ? 1 : type === :HC3 ? 2 : 0
    M = A * (Z' * Diagonal(e .^ 2 ./ (1 .- h) .^ q) * Z) * A
    factor = type === :HC1 ? n / (n - k) : type === :hdm ? n / (n - s - 1) : 1.0
    M[2, 2] * factor
  end
  (; estimate = b[2], variance = V, h, v, w, h_W = diag(P), k, s)
end

@testset "shared treatment-effect final stage" begin
  X, y, d = design()
  n, p = size(X)
  maxima = Dict(method => zeros(4) for method in METHODS)

  @testset "dense references, score, influence, and intervals" begin
    for (Sy, Sd, forced) in (([1, 3], [3, 5], [7]), (Int[], Int[], Int[]))
      fy, fd = record(X, y, Sy), record(X, d, Sd)
      for method in METHODS, type in VCOVS
        ref = dense(y, d, X, fy, fd, method, forced, type)
        originals = (copy(ref.v), copy(ref.w), copy(ref.h_W))
        core = HDM._effect_final(ref.v, ref.w, ref.h_W, ref.k, ref.s, type, method;
          target_norm = norm(d .- mean(d)), outcome_norm = norm(y .- mean(y)))
        @test (ref.v, ref.w, ref.h_W) == originals
        te = double_selection(y, d, X, fy, fd; method, vcov = type, forced, level = 0.9)
        @test core.estimate ≈ ref.estimate atol = 1e-10 rtol = 0
        @test core.variance ≈ ref.variance rtol = 1e-10 atol = 0
        @test te.estimate ≈ ref.estimate atol = 1e-10 rtol = 0
        @test te.se^2 ≈ ref.variance rtol = 1e-10 atol = 0
        @test core.leverage ≈ ref.h atol = 1e-10 rtol = 0
        @test te.max_leverage ≈ maximum(ref.h) atol = 1e-10 rtol = 0
        @test abs(dot(ref.v, core.residuals)) <= 1e-10
        @test core.jacobian == -dot(ref.v, ref.v) / n
        @test core.influence ≈ n .* ref.v .* core.residuals ./ dot(ref.v, ref.v) atol = 1e-12 rtol = 0
        hc0 = HDM._effect_final(ref.v, ref.w, ref.h_W, ref.k, ref.s, :HC0, method)
        @test hc0.variance ≈ sum(abs2, core.influence) / n^2 rtol = 1e-12
        @test te.pvalue ≈ erfc(abs(te.t) / sqrt(2)) atol = 1e-14 rtol = 0
        @test te.t == te.estimate / te.se
        z = -sqrt(2) * erfcinv(2 * 0.95)
        @test collect(te.ci) ≈ [te.estimate - z * te.se, te.estimate + z * te.se] atol = 1e-12 rtol = 0
        @test vec(confint(te; level = 0.9)) ≈ collect(te.ci) atol = 1e-12 rtol = 0
        z80 = -sqrt(2) * erfcinv(2 * 0.9)
        @test vec(confint(te; level = 0.8)) ≈ [te.estimate - z80 * te.se, te.estimate + z80 * te.se] atol = 1e-12 rtol = 0
        @test te.selected_y == Sy && te.selected_d == Sd
        @test te.selected == sort(union(Sy, Sd, method == :doubleselection ? forced : Int[]))
        @test te.forced == (method == :doubleselection ? forced : Int[])
        @test te.rank == ref.k - 1 && te.dof_residual == n - ref.k
        @test te.target_variation ≈ dot(ref.v, ref.v) / sum(abs2, d .- mean(d)) rtol = 1e-12
        @test te.vcov_type == type && te.post
        @test te.refit === (method == :partialling ? true : nothing)
        @test size(coef(te)) == size(stderror(te)) == (1,)
        @test size(vcov(te)) == (1, 1) && vcov(te)[1, 1] == te.se^2
        @test size(confint(te)) == (1, 2) && nobs(te) == n
        text = sprint(show, te)
        @test all(word -> occursin(word, text), (string(method), string(type), "se", "p =", "controls"))
        @test !occursin("non-convergence", text)
        err = maxima[method]
        err[1] = max(err[1], abs(core.estimate - ref.estimate), abs(te.estimate - ref.estimate))
        err[2] = max(err[2], abs(core.variance / ref.variance - 1), abs(te.se^2 / ref.variance - 1))
        err[3] = max(err[3], maximum(abs, core.leverage - ref.h))
        err[4] = max(err[4], abs(dot(ref.v, core.residuals)))
      end
    end
    for method in METHODS
      a, v, h, score = maxima[method]
      println("effect dense $method max_abs_alpha=$a max_relative_variance=$v max_abs_leverage=$h max_abs_score=$score")
    end
  end

  @testset "precomputed fits and RNG" begin
    for method in METHODS, post in (true, false)
      options = CDOptions(opt_tol = 1e-9)
      rng = Xoshiro(118302)
      if method == :doubleselection
        fd = rlasso(X, d, options; post, rng)
        fy = rlasso(X, y, options; post, rng)
      else
        fy = rlasso(X, y, options; post, rng)
        fd = rlasso(X, d, options; post, rng)
      end
      before_y, before_d = deepcopy(fy), deepcopy(fd)
      state = copy(Random.default_rng())
      a = double_selection(y, d, X, fy, fd; method, vcov = :HC1)
      @test copy(Random.default_rng()) == state
      @test same(fy, before_y) && same(fd, before_d)
      b = double_selection(y, d, X; method, post, options, vcov = :HC1, rng = Xoshiro(118302))
      @test same(a, b)
      @test a.nuisance_y === fy && a.nuisance_d === fd
      @test a.post == post && a.refit === (method == :partialling ? post : nothing)
      state = copy(Random.default_rng())
      c = double_selection(y, d, X; method, post, rng = Xoshiro(118303))
      @test copy(Random.default_rng()) == state
      @test same(c, double_selection(y, d, X; method, post, rng = Xoshiro(118303)))
      Random.seed!(118304)
      c = double_selection(y, d, X; method, post)
      Random.seed!(118304)
      @test same(c, double_selection(y, d, X; method, post, rng = Random.default_rng()))
    end

    fy, fd = record(X, y, [1, 3]), record(X, d, [3, 5])
    call(a, b = fd; kwargs...) = double_selection(y, d, X, a, b; kwargs...)
    rejects(() -> call(fy, changed(fd; post = false)), "post")
    rejects(() -> call(changed(fy; hasintercept = false)), "fit_y")
    rejects(() -> call(fy, changed(fd; hasintercept = false)), "fit_d")
    rejects(() -> call(changed(fy; residuals = fy.residuals[1:end-1])), "fit_y"; exception = DimensionMismatch)
    for field in (:x, :lasso, :loadings, :lasso_loadings)
      bad = changed(fy; NamedTuple{(field,)}((getfield(fy, field)[1:end-1],))...)
      rejects(() -> call(bad), "fit_y"; exception = DimensionMismatch)
    end
    other = record(X, y .+ 0.2 .* d, [1, 3])
    rejects(() -> call(other), "fit_y residual consistency")
    rejects(() -> call(fy, record(X, d .+ y, [3, 5])), "fit_d residual consistency")
    rejects(() -> call(record(Float32.(X), Float32.(y), [1, 3])), "fit_y eltype")
    for S in ([3, 1], [1, 1], [p + 1], Int[])
      rejects(() -> call(changed(fy; selected = S)), "fit_y.selected")
    end
    # Partialling continues to ignore forced controls, including invalid ones.
    @test same(call(fy; method = :partialling), call(fy; method = :partialling, forced = [p + 10]))
    @test_throws BoundsError call(fy; forced = [p + 10])
  end

  @testset "input and final-stage guards" begin
    for method in METHODS
      for (yy, dd, quantity) in ((y, ones(n), "d norm"), (ones(n), d, "y norm"),
                                (y, fill(0.1, n), "d norm"), (fill(0.1, n), d, "y norm"))
        rejects(() -> double_selection(yy, dd, X; method, rng = Xoshiro(118305)), quantity)
      end
    end
    rejects(() -> double_selection(y[1:2], d[1:2], X[1:2, :]), "n")
    rejects(() -> double_selection(y, d, zeros(n, 0)), "p")
    rejects(() -> double_selection(y[1:end-1], d, X), "length"; exception = DimensionMismatch)
    rejects(() -> double_selection(y, d, X[1:end-1, :]), "length"; exception = DimensionMismatch)
    for level in (0, 1, -0.1, Inf, NaN)
      rejects(() -> double_selection(y, d, X; level), "level")
    end
    for method in (:invalid, "partialling", nothing)
      rejects(() -> double_selection(y, d, X; method), "method")
    end
    for type in (:invalid, "HC0", nothing)
      rejects(() -> double_selection(y, d, X; vcov = type), "vcov")
    end
    for intercept in (false, 0, nothing)
      rejects(() -> double_selection(y, d, X; intercept), "intercept")
    end
    for field in (:weights, :offset, :refit)
      rejects(() -> double_selection(y, d, X; NamedTuple{(field,)}((true,))...), string(field))
    end
    for value in (Inf, NaN), which in (:y, :d, :X)
      yy, dd, XX = copy(y), copy(d), copy(X)
      which == :y ? (yy[1] = value) : which == :d ? (dd[1] = value) : (XX[1, 1] = value)
      rejects(() -> double_selection(yy, dd, XX), string(which))
    end
    # Saturation is checked for every variance, independently of residual size.
    smallrng = Xoshiro(118306)
    W, yy, dd = randn(smallrng, 6, 4), randn(smallrng, 6), randn(smallrng, 6)
    fy, fd = record(W, yy), record(W, dd)
    for type in VCOVS
      rejects(() -> double_selection(yy, dd, W, fy, fd; forced = collect(1:4), vcov = type), "k=")
      for method in METHODS
        rejects(() -> HDM._effect_final(dd, yy, fill(1/6, 6), 6, 0, type, method), "k=")
      end
    end
    te = double_selection(yy, dd, W, fy, fd; forced = [1, 2, 3], vcov = :HC0)
    @test te.dof_residual == 1 && isfinite(te.se)
    A = repeat(W[:, 1:1], 1, 5)
    rejects(() -> double_selection(yy, dd, A, record(A, yy), record(A, dd); forced = collect(1:5)), "n-s-1=0")

    A = reshape(copy(d), :, 1)
    rejects(() -> double_selection(y, d, A, record(A, y), record(A, d); forced = [1]), "target norm")
    rejects(() -> double_selection(y, d, A, record(A, y), record(A, d, [1]); method = :partialling), "target norm")
    for method in METHODS
      yy = 2 .* d .+ 3
      rejects(() -> double_selection(yy, d, X, record(X, yy), record(X, d); method), "residual norm")
    end

    A = zeros(n, 1); A[1, 1] = 1
    for type in (:HC2, :HC3)
      rejects(() -> double_selection(y, d, A, record(A, y), record(A, d); forced = [1], vcov = type), "max leverage")
    end
    # An intercept alias is counted nominally but contributes no basis rank.
    for column in (ones(n), 1e10 .+ 0.1 .* X[:, 1])
      A = reshape(column, :, 1)
      fy, fd = record(A, y), record(A, d)
      a = double_selection(y, d, A, fy, fd; forced = [1], vcov = :hdm)
      b = double_selection(y, d, A, fy, fd; forced = [1], vcov = :HC0)
      @test a.rank == 1 && a.selected == [1] && a.dof_residual == n - 2
      @test a.se^2 ≈ b.se^2 * n / (n - 2) rtol = 1e-12
    end

    te = @test_logs (:warn, r"coordinate descent did not converge") (:warn, r"coordinate descent did not converge") double_selection(
      y, d, X; options = CDOptions(max_iter = 0), rng = Xoshiro(118307))
    @test !te.nuisance_y.solver_converged && !te.nuisance_d.solver_converged
    @test occursin("nuisance non-convergence: y, d", sprint(show, te))
    @test_throws ArgumentError confint(te; level = 0)
  end

  @testset "original-unit transformations" begin
    a, b, c = -2.5, 3.7, -0.4
    for method in METHODS, type in VCOVS, post in (true, false)
      fy, fd = record(X, y, [1, 3]; post), record(X, d, [3, 5]; post)
      base = double_selection(y, d, X, fy, fd; method, vcov = type, forced = [7])
      yy = a .* y .+ b
      r = double_selection(yy, d, X, record(X, yy, [1, 3]; post), fd; method, vcov = type, forced = [7])
      @test r.estimate ≈ a * base.estimate atol = 1e-10 rtol = 0
      @test r.se ≈ abs(a) * base.se rtol = 1e-10
      dd = c .* d
      r = double_selection(y, dd, X, fy, record(X, dd, [3, 5]; post); method, vcov = type, forced = [7])
      @test r.estimate ≈ base.estimate / c atol = 1e-10 rtol = 0
      @test r.se ≈ base.se / abs(c) rtol = 1e-10
      @test r.target_variation ≈ base.target_variation rtol = 1e-10
      @test r.max_leverage ≈ base.max_leverage atol = 1e-10 rtol = 0
    end
  end
end

end
