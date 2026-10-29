module MultitaskSpeedTest

using HighDimensionalModels
using LinearAlgebra, Random, SparseArrays, Test
using ProximalBase: GroupNormL2
import ProximalSolvers as PS

const HDM = HighDimensionalModels

# draws with mean zero and variance one: centered uniforms, so that the test
# data come from rand alone, whose streams are the same across Julia versions
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

# q tasks with ns[t] observations of p predictors, the columns on different
# scales in each task, the rows `support` of the coefficients nonzero, and the
# intercept t in task t
function make_tasks(rng, ns, p; support = 1:4)
  q = length(ns)
  B = zeros(p, q)
  B[support, :] .= centered_rand(rng, length(support), q)
  Xs = [centered_rand(rng, n, p) .* (1 .+ 2 .* rand(rng, p))' for n in ns]
  ys = [t .+ Xs[t] * B[:, t] .+ 0.5 .* centered_rand(rng, ns[t]) for t in 1:q]
  Xs, ys
end

# Momentum on the stacked design of task-specific designs: on by default with
# the random order, off with momentum = false or the cyclic order. Every solve
# draws its order from a generator of its own with the same seed on both sides
# of a comparison, so that equal options give equal passes.
@testset "multitask momentum" begin
  # near interpolation: task 1 has fewer observations than predictors, and
  # ∑ₜ nₜ ≥ pq, so that the automatic grid goes down to 1e-4 λmax, where
  # coordinate descent on the rows takes many passes
  ns, p, nλ = (15, 40, 60), 20, 30
  Xs, ys = make_tasks(Xoshiro(1431), ns, p)
  @test sum(ns) >= p * length(ns) && minimum(ns) < p
  path(o = CDOptions()) = MultitaskLassoPath(Xs, ys, o; nλ, rng = Xoshiro(1))
  on, off = path(), path(CDOptions(; momentum = false))
  @test all(on.converged) && all(off.converged)
  @test on.λpath == off.λpath
  @test on.λpath[end] ≈ 1e-4 * on.λpath[1] rtol = 1e-10

  @testset "the options" begin
    # the default options take the passes of momentum = true, and those of
    # momentum = false differ; the cyclic order takes the same passes with and
    # without momentum = false, so it runs without momentum
    @test path(CDOptions(; momentum = true)).iterations == on.iterations
    @test on.iterations != off.iterations
    @test path(CDOptions(; order = :cyclic)).iterations ==
          path(CDOptions(; order = :cyclic, momentum = false)).iterations
    # single fits, warm started and by continuation from λmax (_solve!)
    λ = 0.01 * on.λpath[1]
    fit(o) = multitask_lasso(Xs, ys, λ, o; rng = Xoshiro(2))
    for warm_start in (true, false)
      o = CDOptions(; warm_start)
      s = fit(o)
      @test s.converged
      @test fit(CDOptions(o; momentum = true)).iterations == s.iterations
      @test fit(CDOptions(o; momentum = false)).iterations != s.iterations
      @test fit(CDOptions(o; order = :cyclic)).iterations ==
            fit(CDOptions(o; order = :cyclic, momentum = false)).iterations
    end
    # the paths of the folds of cross-validation take the same options; the
    # fold levels differ by the tolerance with momentum = false, and so the
    # errors
    folds = [mod1.(1:n, 3) for n in ns]
    cv(o = CDOptions()) = CVMultitaskLassoPath(Xs, ys, o; folds, nλ = 10, rng = Xoshiro(3))
    c = cv()
    @test isequal(cv(CDOptions(; momentum = true)).cvm, c.cvm)
    @test !isequal(cv(CDOptions(; momentum = false)).cvm, c.cvm)
  end

  @testset "fewer passes" begin
    # a loose bound: the ratio is 0.14 to 0.33 over a dozen data seeds and two
    # orders
    @test sum(on.iterations) < 0.8 * sum(off.iterations)
  end

  @testset "the same solutions" begin
    # the objective at every level of the path, with a tight duality gap
    tight = CDOptions(; gaptol = 1e-11)
    a, b = path(tight), path(CDOptions(tight; momentum = false))
    @test all(a.converged) && all(b.converged) && a.λpath == b.λpath
    d = HDM._MultitaskData(Xs, ys; intercept = true, standardize_x = true, task_weights = :equal)
    objective(P, k) = HDM._task_objective(d, P.Bpath[k], P.β0path[k], P.λpath[k], ones(p))
    @test all(k -> isapprox(objective(a, k), objective(b, k); rtol = 1e-8), eachindex(a.λpath))
    # a single fit on well-conditioned designs (more observations than
    # predictors in every task) with unequal weights, where a block step
    # minimizes a majorization: the same coefficients
    Xw, yw = make_tasks(Xoshiro(1432), (30, 60, 90), 15)
    tw = [0.5, 1.0, 2.0]
    λ = 0.01 * MultitaskLassoPath(Xw, yw; nλ = 1, task_weights = tw).λpath[1]
    o = CDOptions(; gaptol = 1e-12)
    s1 = multitask_lasso(Xw, yw, λ, o; task_weights = tw, rng = Xoshiro(4))
    s0 = multitask_lasso(Xw, yw, λ, CDOptions(o; momentum = false); task_weights = tw, rng = Xoshiro(4))
    @test s1.converged && s0.converged && s1.iterations < s0.iterations
    @test Matrix(s1.B) ≈ Matrix(s0.B) atol = 1e-8
    @test s1.intercept ≈ s0.intercept atol = 1e-8
  end

  @testset "a shared design: the passes of MultiLassoPath" begin
    # the same matrix for every task and equal weights take the model of
    # multi_lasso, with its options unchanged: momentum stays off by default
    # and on with momentum = true
    n, q = 30, 3
    rng = Xoshiro(1433)
    X = centered_rand(rng, n, p) .* (1 .+ 2 .* rand(rng, p))'
    Y = X[:, 1:3] * centered_rand(rng, 3, q) .+ 0.5 .* centered_rand(rng, n, q)
    ysh = [Y[:, t] for t in 1:q]
    @test HDM._MultitaskData(fill(X, q), ysh; intercept = true, standardize_x = true, task_weights = :equal,
                             shared = true).Z === nothing
    for o in (CDOptions(), CDOptions(; momentum = false), CDOptions(; momentum = true))
      mp = MultiLassoPath(X, Y, o; nλ, rng = Xoshiro(5))
      mt = MultitaskLassoPath(fill(X, q), ysh, o; nλ, rng = Xoshiro(5))
      @test mt.λpath == mp.λpath
      @test mt.iterations == mp.iterations
    end
    @test MultitaskLassoPath(fill(X, q), ysh; nλ, rng = Xoshiro(5)).iterations ==
          MultitaskLassoPath(fill(X, q), ysh, CDOptions(; momentum = false); nλ, rng = Xoshiro(5)).iterations
  end
end

# The R factors of the tall blocks of the stacked design in paths: the model of
# _compressed_model is that of Z up to rounding, and so are the paths.
@testset "multitask R factors" begin
  @testset "the same model" begin
    # tasks with nₜ < p, nₜ = p, nₜ = p + 1 and nₜ > p with a duplicated
    # column (a rank-deficient block), a constant design, which has no fitted
    # column with an intercept, and a sparse design with 30% nonzeros, dense
    # in Z with an intercept only
    rng = Xoshiro(1434)
    p = 6
    Xs = Any[centered_rand(rng, n, p) .* (1 .+ 2 .* rand(rng, p))' for n in (4, 6, 7, 20)]
    Xs[4][:, 3] = Xs[4][:, 2]
    push!(Xs, ones(9, p))
    push!(Xs, sparse(centered_rand(rng, 30, p) .* (rand(rng, 30, p) .< 0.3)))
    ys = [centered_rand(rng, size(X, 1)) for X in Xs]
    θ = centered_rand(Xoshiro(1435), length(Xs) * p)
    for T in (Float64, Float32), intercept in (true, false), standardize_x in (true, false),
        task_weights in (:equal, :observations)
      d = HDM._MultitaskData([T.(X) for X in Xs], [T.(y) for y in ys]; intercept, standardize_x, task_weights)
      _, cols = HDM._task_blocks(d)
      @test length.(cols) == [p, p, p, p, intercept ? 0 : p, p]
      which = HDM._compressed_tasks(d)
      @test which == [false, false, true, true, true, intercept]
      m0, x, groups, w = HDM._multitask_problem(d, ones(T, p))
      m1 = HDM._compressed_model(d, which)
      # the model of Z, with pₜ rows for a compressed task, nₜ for the others,
      # and the extra row, without entries
      @test typeof(m1) == typeof(m0) && m1.s == m0.s && !m1.intercept
      @test size(m1.X) == (sum(which[t] ? length(cols[t]) : d.nobs[t] for t in eachindex(cols)) + 1, size(d.Z, 2))
      @test iszero(m1.X[end, :]) && !iszero(m1.y[end])
      rtol = T === Float64 ? 1e-12 : 1e-5
      same(a, b) = norm(a - b) <= rtol * norm(b)
      g = GroupNormL2(one(T), groups, w)
      λmax = PS.lambda_max(m0, g, x)
      @test same(PS.lambda_max(m1, g, x), λmax)
      # at a point with every row nonzero
      for k in eachindex(x)
        x[k] = T(θ[k])
      end
      PS.initialize!(m0, x)
      PS.initialize!(m1, x)
      @test same(PS.loss_value(m1, x), PS.loss_value(m0, x))
      @test same(PS.gradient!(zeros(T, length(x)), m1, x), PS.gradient!(zeros(T, length(x)), m0, x))
      @test same(PS.response_scale(m1, x), PS.response_scale(m0, x))
      @test same([PS.coordinate_scale(m1, x, k) for k in eachindex(x)],
                 [PS.coordinate_scale(m0, x, k) for k in eachindex(x)])
      @test same([PS._groupcurvature!(m1, x, k) for k in eachindex(groups)],
                 [PS._groupcurvature!(m0, x, k) for k in eachindex(groups)])
      h = GroupNormL2(T(0.3) * λmax, groups, w)
      gap0, gap1 = PS.duality_gap(m0, h, x), PS.duality_gap(m1, h, x)
      @test same(gap1.primal, gap0.primal) && same(gap1.dual, gap0.dual) && same(gap1.gap, gap0.gap)
    end
  end

  @testset "the same paths" begin
    # one task with fewer observations than predictors, left alone, and one
    # with more, replaced by its R factor; the cyclic order without momentum
    # makes the passes deterministic
    ns, p, nλ = (15, 60), 20, 20
    Xs, ys = make_tasks(Xoshiro(1436), ns, p)
    d = HDM._MultitaskData(Xs, ys; intercept = true, standardize_x = true, task_weights = :equal)
    @test HDM._compressed_tasks(d) == [false, true]
    o = CDOptions(; order = :cyclic, momentum = false, gaptol = 1e-10)
    a = HDM._multitask_path(Xs, ys, nothing, o; nλ)
    b = HDM._multitask_path(Xs, ys, nothing, o; nλ, compress = false)
    @test a.λpath == b.λpath && all(a.converged) && all(b.converged)
    @test all(abs.(a.iterations .- b.iterations) .<= 1)
    objective(P, k) = HDM._task_objective(d, P.Bpath[k], P.β0path[k], P.λpath[k], ones(p))
    @test all(k -> isapprox(objective(a, k), objective(b, k); rtol = 1e-10), eachindex(a.λpath))
  end
end

end
