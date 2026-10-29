module MultitaskSpeedTest

using HighDimensionalModels
using Random, Test

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

end
