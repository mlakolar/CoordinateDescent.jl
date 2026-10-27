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

function dense_partial(u, v, type)
  n = length(u)
  Z = [ones(n) v]
  b = Z \ u
  e = u - Z * b
  A = inv(Z' * Z)
  h = diag(Z * A * Z')
  V = if type in (:hdm, :homoskedastic)
    sum(abs2, e) / (n - 2) * A[2, 2]
  else
    q = type === :HC2 ? 1 : type === :HC3 ? 2 : 0
    M = A * (Z' * Diagonal(e .^ 2 ./ (1 .- h) .^ q) * Z) * A
    M[2, 2] * (type === :HC1 ? n / (n - 2) : 1.0)
  end
  (; estimate = b[2], variance = V, residuals = e, h)
end

function kkt(fit, response, X)
  Xc = X .- mean(X; dims = 1)
  residual = (response .- mean(response)) - Xc * fit.lasso
  g = Xc' * residual / length(response)
  penalty = fit.lasso_λ0 .* fit.lasso_loadings
  S, off = fit.selected, setdiff(axes(X, 2), fit.selected)
  on_error = isempty(S) ? 0.0 : maximum(abs, g[S] - penalty[S] .* sign.(fit.lasso[S]))
  off_error = isempty(off) ? 0.0 : max(0.0, maximum(abs.(g[off]) - penalty[off]))
  max(on_error, off_error)
end

function rescaled(fit, a, b = 0.0)
  changed(fit; x = a .* fit.x, intercept = a * fit.intercept + b,
    residuals = a .* fit.residuals, σ = abs(a) * fit.σ, loadings = abs(a) .* fit.loadings,
    lasso = a .* fit.lasso, lasso_intercept = a * fit.lasso_intercept + b,
    lasso_loadings = abs(a) .* fit.lasso_loadings)
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
        force = method == :doubleselection ? forced : Int[]
        te = double_selection(y, d, X, fy, fd; method, vcov = type, forced = force, level = 0.9)
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
    rejects(() -> call(fy; method = :partialling, forced = [p + 10]), "native hdm ignores I3")
    rejects(() -> double_selection(y, d, X; method = :partialling, forced = [1], rng = Xoshiro(118308)), "forced")
    for refit in (true, false)
      rejects(() -> call(fy; refit), "refit")
      rejects(() -> double_selection(y, d, X; refit, rng = Xoshiro(118309)), "refit")
    end
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
      forced = method == :doubleselection ? [7] : Int[]
      base = double_selection(y, d, X, fy, fd; method, vcov = type, forced)
      yy = a .* y .+ b
      r = double_selection(yy, d, X, record(X, yy, [1, 3]; post), fd; method, vcov = type, forced)
      @test r.estimate ≈ a * base.estimate atol = 1e-10 rtol = 0
      @test r.se ≈ abs(a) * base.se rtol = 1e-10
      dd = c .* d
      r = double_selection(y, dd, X, fy, record(X, dd, [3, 5]; post); method, vcov = type, forced)
      @test r.estimate ≈ base.estimate / c atol = 1e-10 rtol = 0
      @test r.se ≈ base.se / abs(c) rtol = 1e-10
      @test r.target_variation ≈ base.target_variation rtol = 1e-10
      @test r.max_leverage ≈ base.max_leverage atol = 1e-10 rtol = 0
    end
  end

  @testset "controlled partialling refit modes" begin
    options = CDOptions(opt_tol = 1e-10)
    direct_errors = Dict(true => zeros(3), false => zeros(3))
    for post in (true, false), empty in (false, true)
      c = empty ? 100.0 : post ? 1.1 : 0.5
      fy = rlasso(X, y, options; post, c, rng = Xoshiro(118401))
      fd = rlasso(X, d, options; post, c, rng = Xoshiro(118402))
      before_y, before_d = deepcopy(fy), deepcopy(fd)
      if empty
        @test isempty(fy.selected) && isempty(fd.selected)
      else
        @test fy.selected != fd.selected
      end
      # The same U2 1e-8 score allowance applies to 1e-10 CD steps.
      kkt_error = max(kkt(fy, y, X), kkt(fd, d, X))
      @test kkt_error <= 1e-8
      refit = !post
      u, v = if refit
        Zy, Zd = [ones(n) X[:, fy.selected]], [ones(n) X[:, fd.selected]]
        (y - Zy * (Zy \ y), d - Zd * (Zd \ d))
      else
        (y .- fy.lasso_intercept .- X * fy.lasso, d .- fd.lasso_intercept .- X * fd.lasso)
      end
      used_u = HDM._partialling_residuals(y, X, fy, refit)
      used_v = HDM._partialling_residuals(d, X, fd, refit)
      @test used_u ≈ u atol = 1e-10 rtol = 0
      @test used_v ≈ v atol = 1e-10 rtol = 0
      if !refit
        @test used_u == u && used_v == v
      end
      @test HDM._partialling_residuals(y, X, fy, post) === fy.residuals
      @test HDM._partialling_residuals(d, X, fd, post) === fd.residuals
      state = copy(Random.default_rng())
      for type in VCOVS
        te = double_selection(y, d, X, fy, fd; method = :partialling, refit, vcov = type)
        native = double_selection(y, d, X, fy, fd; method = :partialling, vcov = type)
        @test same(native, double_selection(y, d, X, fy, fd; method = :partialling, refit = post, vcov = type))
        @test te.nuisance_y === native.nuisance_y === fy
        @test te.nuisance_d === native.nuisance_d === fd
        @test te.refit === refit && te.post === post
        @test te.selected_y == fy.selected && te.selected_d == fd.selected
        @test te.nuisance_y.lasso_λ0 == fy.lasso_λ0 && te.nuisance_d.lasso_λ0 == fd.lasso_λ0
        @test te.nuisance_y.lasso_loadings === fy.lasso_loadings
        @test te.nuisance_d.lasso_loadings === fd.lasso_loadings
        ref = dense_partial(u, v, type)
        @test te.estimate ≈ ref.estimate atol = 1e-10 rtol = 0
        @test te.se^2 ≈ ref.variance rtol = 1e-10 atol = 0
        @test te.max_leverage ≈ maximum(ref.h) atol = 1e-10 rtol = 0
        layout = HDM._effect_layout(y, d, X, fy, fd, :partialling, Int[], refit)
        core = HDM._effect_final(layout.v, layout.w, layout.h_W, 2, 0, type, :partialling)
        @test core.residuals ≈ ref.residuals atol = 1e-10 rtol = 0
        err = direct_errors[post]
        err[1] = max(err[1], abs(te.estimate - ref.estimate))
        err[2] = max(err[2], abs(te.se^2 / ref.variance - 1))
        err[3] = max(err[3], maximum(abs, core.leverage - ref.h))
        if !empty
          @test abs(te.estimate - native.estimate) > 1e-8
        end
        # Unit changes transform the same fitted record, keeping its support
        # and penalty problem fixed apart from the corresponding unit changes.
        a, b, scale_d = -2.5, 3.7, -0.4
        ty = double_selection(a .* y .+ b, d, X, rescaled(fy, a, b), fd;
          method = :partialling, refit, vcov = type)
        td = double_selection(y, scale_d .* d, X, fy, rescaled(fd, scale_d);
          method = :partialling, refit, vcov = type)
        @test ty.estimate ≈ a * te.estimate atol = 1e-10 rtol = 0
        @test ty.se ≈ abs(a) * te.se rtol = 1e-10
        @test td.estimate ≈ te.estimate / scale_d atol = 1e-10 rtol = 0
        @test td.se ≈ te.se / abs(scale_d) rtol = 1e-10
      end
      @test copy(Random.default_rng()) == state
      @test same(fy, before_y) && same(fd, before_d)
      if !empty
        native = double_selection(y, d, X, fy, fd; method = :partialling)
        other = double_selection(y, d, X, fy, fd; method = :partialling, refit)
        println("partialling controlled post=$post selected_y=$(fy.selected) selected_d=$(fd.selected) KKT=$kkt_error native_alpha=$(native.estimate) refit_$(refit)_alpha=$(other.estimate)")
      end
    end
    for post in (true, false)
      a, v, h = direct_errors[post]
      println("partialling non-native post=$post max_abs_alpha=$a max_relative_variance=$v max_abs_leverage=$h")
    end
  end

  @testset "equal-support identity and degrees of freedom" begin
    S = [1, 3, 5]
    r = size(HDM._effect_basis(X, S), 2)
    expected_ratio = (n - 2) / (n - r - 2)
    for post in (true, false)
      fy, fd = record(X, y, S; post), record(X, d, S; post)
      po = HDM._effect_layout(y, d, X, fy, fd, :partialling, Int[], true)
      ds = HDM._effect_layout(y, d, X, fy, fd, :doubleselection, Int[], nothing)
      cp = HDM._effect_final(po.v, po.w, po.h_W, po.k, po.s, :HC0, :partialling)
      cd = HDM._effect_final(ds.v, ds.w, ds.h_W, ds.k, ds.s, :HC0, :doubleselection)
      @test cp.estimate ≈ cd.estimate atol = 1e-10 rtol = 0
      @test cp.residuals ≈ cd.residuals atol = 1e-10 rtol = 0
      @test cp.variance ≈ cd.variance rtol = 1e-10 atol = 0
      for type in VCOVS
        a = double_selection(y, d, X, fy, fd; method = :partialling, refit = true, vcov = type)
        b = double_selection(y, d, X, fy, fd; method = :doubleselection, vcov = type)
        @test a.estimate ≈ b.estimate atol = 1e-10 rtol = 0
        ratio = b.se^2 / a.se^2
        if type in (:homoskedastic, :HC1)
          @test ratio ≈ expected_ratio rtol = 1e-10 atol = 0
          println("partialling identity post=$post vcov=$type DS_over_PO=$ratio expected=$expected_ratio r=$r")
        elseif type === :HC0
          @test ratio ≈ 1 rtol = 1e-10 atol = 0
        else
          @test !isapprox(b.se^2, a.se^2; rtol = 1e-10, atol = 0)
        end
      end
    end
  end

  @testset "refit bounds and weak target diagnostic" begin
    rng = Xoshiro(118403)
    W, yy, dd = randn(rng, 6, 5), randn(rng, 6), randn(rng, 6)
    for post in (true, false)
      fy, fd = record(W, yy, collect(1:5); post), record(W, dd; post)
      rejects(() -> double_selection(yy, dd, W, fy, fd; method = :partialling, refit = true), "fit_y refit")
      rejects(() -> double_selection(yy, dd, W, record(W, yy; post), record(W, dd, collect(1:5); post);
        method = :partialling, refit = true), "fit_d refit")
      if post
        rejects(() -> double_selection(yy, dd, W, fy, fd; method = :partialling), "|S|=5 and n=6")
      end
    end
    Z = [ones(n) X[:, 1]]
    noise = randn(rng, n)
    v = noise - Z * (Z \ noise)
    v .*= norm(X[:, 1] .- mean(X[:, 1])) / norm(v)
    delta = 1e-5
    dd = X[:, 1] .+ delta .* v
    yy = 0.5 .* dd .+ X[:, 1] .+ randn(rng, n)
    te = double_selection(yy, dd, X, record(X, yy, [1]), record(X, dd, [1]); method = :partialling)
    @test HDM._effect_tau(Float64)^2 < te.target_variation < 1e-8
    # Cancellation in a 1e-5 residual permits a 1e-8 relative diagnostic error.
    @test te.target_variation ≈ delta^2 / (1 + delta^2) rtol = 1e-8 atol = 0
    println("partialling weak target_variation=$(te.target_variation) expected=$(delta^2/(1+delta^2))")
  end
end

end
