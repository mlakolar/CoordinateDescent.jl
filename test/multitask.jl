module MultitaskLassoTest

using HighDimensionalModels
using LinearAlgebra, Random, SparseArrays, Statistics, Test

const TIGHT = CDOptions(; opt_tol = 1e-10, gaptol = 0)
const EXACT = CDOptions(; opt_tol = 1e-14, gaptol = 0)

# draws with mean zero and variance one: centered uniforms, so that the test
# data come from rand alone, whose streams are the same across Julia versions
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

# q tasks with ns[t] observations of p predictors, the columns on different
# scales and centers in each task, the rows `support` of the coefficients
# nonzero, and the intercept t in task t
function make_tasks(rng, ns, p; support = 1:4)
  q = length(ns)
  B = zeros(p, q)
  B[support, :] .= centered_rand(rng, length(support), q)
  Xs = [centered_rand(rng, n, p) .* (1 .+ 2 .* rand(rng, p))' .+ 0.5 .* centered_rand(rng, p)' for n in ns]
  ys = [t .+ Xs[t] * B[:, t] .+ centered_rand(rng, ns[t]) for t in 1:q]
  Xs, ys
end

# the weights cₜ of the keyword task_weights for tasks of sizes ns
weight_values(tw, ns) = tw === :equal ? ones(length(ns)) :
  tw === :observations ? length(ns) .* collect(ns) ./ sum(ns) : float.(collect(tw))

# the scales sₜⱼ of the penalty: the population standard deviation of column j
# within task t, centered with an intercept (the root mean square without one),
# or one without standardization; zero for a zero column and, with an
# intercept, for a constant one
function scales(Xs, intercept, standardize_x)
  p = size(Xs[1], 2)
  S = zeros(p, length(Xs))
  for t in eachindex(Xs), j in 1:p
    x = Vector{Float64}(Xs[t][:, j])
    (intercept ? all(==(x[1]), x) : all(iszero, x)) && continue
    S[j, t] = standardize_x ? sqrt(sum(abs2, x .- (intercept ? mean(x) : 0.0)) / length(x)) : 1.0
  end
  S
end

# ∑ₜ cₜ ‖yₜ - β₀ₜ - Xₜbₜ‖² / (2nₜ) + λ ∑ⱼ ωⱼ ‖(s₁ⱼ b₁ⱼ, …, s_qⱼ b_qⱼ)‖
function objective(Xs, ys, β0, B, λ, c, ω, S)
  B = Matrix(B)
  loss = sum(c[t] * sum(abs2, ys[t] .- β0[t] .- Xs[t] * B[:, t]) / (2 * length(ys[t])) for t in eachindex(ys))
  loss + λ * sum(ω[j] * norm(S[j, :] .* B[j, :]) for j in axes(B, 1))
end

# The largest violation of the optimality conditions of the objective at
# (β0, B), from the gradient ∇ₜⱼ = -cₜ Xₜⱼ'rₜ / nₜ of the loss on the original
# scale: ‖(∇ₜⱼ / sₜⱼ)ₜ‖ ≤ λωⱼ over the tasks with sₜⱼ > 0 on a zero row j,
# ∇ⱼ + λωⱼ Sⱼ² bⱼ / ‖Sⱼbⱼ‖ = 0 on a nonzero one, bₜⱼ = 0 and ∇ₜⱼ = 0 where
# sₜⱼ = 0 (the centered column is zero), and ∑ᵢ rₜᵢ = 0 with an intercept.
function kkt(Xs, ys, β0, B, λ, c, ω, S; intercept = true)
  B = Matrix(B)
  p, q = size(B)
  G = zeros(p, q)
  res = 0.0
  for t in 1:q
    r = ys[t] .- β0[t] .- Xs[t] * B[:, t]
    G[:, t] = -c[t] .* (Xs[t]' * r) ./ length(r)
    intercept && (res = max(res, c[t] * abs(sum(r)) / length(r)))
  end
  for j in 1:p
    on = S[j, :] .> 0
    any(!iszero, B[j, .!on]) && return Inf
    res = max(res, maximum(abs, G[j, .!on]; init = 0.0))
    θ = S[j, :] .* B[j, :]
    res = max(res, iszero(θ) ? max(norm(G[j, on] ./ S[j, on]) - λ * ω[j], 0.0) :
                               norm(G[j, :] .+ λ * ω[j] .* S[j, :] .^ 2 .* B[j, :] ./ norm(θ)))
  end
  res
end

# the smallest λ at which B = 0 is optimal, maxⱼ ‖(∇ₜⱼ / sₜⱼ)ₜ‖ / ωⱼ at B = 0
# with the intercepts at the means of the responses
function lambda_max(Xs, ys, c, ω, S; intercept = true)
  maximum(axes(S, 1)) do j
    g = [S[j, t] > 0 ? c[t] * dot(Xs[t][:, j], ys[t] .- intercept * mean(ys[t])) / length(ys[t]) / S[j, t] : 0.0
         for t in eachindex(ys)]
    norm(g) / ω[j]
  end
end

# The minimizer of the objective by accelerated proximal gradient (FISTA) in
# the coordinates θₜⱼ = sₜⱼ bₜⱼ of the standardized columns, with the
# intercepts as free variables rather than centered out. The step is 1/L, L the
# largest curvature of the loss of a task over (β₀ₜ, θₜ); the prox of the rows
# is θⱼ ← (1 - λωⱼ / (L‖θⱼ‖))₊ θⱼ; the momentum restarts when a step goes
# against it (O'Donoghue and Candès, 2015). It stops once the gradient mapping
# at the extrapolated point is below tol, which puts the new iterate within
# 2 tol of stationarity. Entries with sₜⱼ = 0 have zero columns and stay zero.
function reference_fit(Xs, ys, λ, c, ω, S; intercept = true, tol = 1e-10, max_iter = 50_000)
  p, q = size(S)
  Zs = [Matrix{Float64}(Xs[t]) .* [S[j, t] > 0 ? 1 / S[j, t] : 0.0 for j in 1:p]' for t in 1:q]
  L = maximum(c[t] * opnorm(intercept ? [ones(size(Zs[t], 1)) Zs[t]] : Zs[t])^2 / size(Zs[t], 1) for t in 1:q)
  function gradient(a, Θ)
    ga, GΘ = zeros(q), zeros(p, q)
    for t in 1:q
      r = ys[t] .- a[t] .- Zs[t] * Θ[:, t]
      GΘ[:, t] = -c[t] .* (Zs[t]' * r) ./ length(r)
      ga[t] = intercept ? -c[t] * sum(r) / length(r) : 0.0
    end
    ga, GΘ
  end
  function prox(Θ)
    P = copy(Θ)
    for j in 1:p
      nj = norm(Θ[j, :])
      P[j, :] .*= nj > 0 ? max(1 - λ * ω[j] / (L * nj), 0.0) : 0.0
    end
    P
  end
  unscale(Θ) = [S[j, t] > 0 ? Θ[j, t] / S[j, t] : 0.0 for j in 1:p, t in 1:q]
  a, Θ = zeros(q), zeros(p, q)
  ya, yΘ, τ = copy(a), copy(Θ), 1.0
  for k in 1:max_iter
    ga, GΘ = gradient(ya, yΘ)
    an, Θn = ya .- ga ./ L, prox(yΘ .- GΘ ./ L)
    L * sqrt(sum(abs2, ya .- an) + sum(abs2, yΘ .- Θn)) < tol &&
      return (; β0 = an, B = unscale(Θn), converged = true, iterations = k)
    if dot(ya .- an, an .- a) + dot(yΘ .- Θn, Θn .- Θ) > 0
      τ = 1.0
      ya, yΘ = copy(an), copy(Θn)
    else
      τn = (1 + sqrt(1 + 4τ^2)) / 2
      ya, yΘ = an .+ ((τ - 1) / τn) .* (an .- a), Θn .+ ((τ - 1) / τn) .* (Θn .- Θ)
      τ = τn
    end
    a, Θ = an, Θn
  end
  (; β0 = a, B = unscale(Θ), converged = false, iterations = max_iter)
end

# argmin over b of ∑ₜ cₜ (bₜ - b̂ₜ)² / 2 + κ‖b‖: zero when ‖c ∘ b̂‖ ≤ κ, and
# otherwise bₜ = cₜ b̂ₜ / (cₜ + κ/r), where r = ‖b‖ solves
# ∑ₜ (cₜ b̂ₜ / (cₜ r + κ))² = 1, whose left side decreases from ‖c ∘ b̂‖² / κ²
# at r = 0 to below one at r = ‖b̂‖ (bisection)
function weighted_threshold(b̂, c, κ)
  norm(c .* b̂) <= κ && return zero(b̂)
  lo, hi = 0.0, norm(b̂)
  for _ in 1:200
    mid = (lo + hi) / 2
    sum(abs2, c .* b̂ ./ (c .* mid .+ κ)) > 1 ? (lo = mid) : (hi = mid)
  end
  r = (lo + hi) / 2
  c .* b̂ ./ (c .+ κ / r)
end

# The held-out errors of the folds of cv, recomputed from fold paths fitted by
# MultitaskLassoPath on the training rows alone, with options and the keywords
# given: e_k = ∑ₜ (cₜ/nₜ) SSEₜₖ / w_k, w_k = ∑ₜ cₜ mₜₖ / nₜ, at the levels the
# path of fold k reached. Returns, at each level, their mean weighted by the w_k
# over the folds that reached it, its standard error (NaN for one fold), and the
# held-out mean squared error of each task over those folds, all NaN at a level
# no fold reached; and the number of levels each fold path reached.
function cv_recompute(cv, Xs, ys, c, tw, options = TIGHT; kwargs...)
  q, L = length(Xs), length(cv.λpath)
  K = maximum(maximum, cv.folds)
  ns = length.(ys)
  errs, wk, Lk = fill(NaN, K, L), zeros(K), zeros(Int, K)
  sse, mt = zeros(q, K, L), zeros(q, K)
  for k in 1:K
    train = [findall(!=(k), f) for f in cv.folds]
    test = [findall(==(k), f) for f in cv.folds]
    fp = MultitaskLassoPath([Xs[t][train[t], :] for t in 1:q], [ys[t][train[t]] for t in 1:q], cv.λpath, options;
                            task_weights = tw, kwargs...)
    Ŷ = predict(fp, [Xs[t][test[t], :] for t in 1:q])
    Lk[k] = length(fp.λpath)
    for t in 1:q
      mt[t, k] = length(test[t])
      sse[t, k, 1:Lk[k]] = sum(abs2, ys[t][test[t]] .- Ŷ[t]; dims = 1)
    end
    wk[k] = sum(c[t] * mt[t, k] / ns[t] for t in 1:q)
    errs[k, 1:Lk[k]] = [sum(c[t] / ns[t] * sse[t, k, j] for t in 1:q) / wk[k] for j in 1:Lk[k]]
  end
  cvm, cvse, cvm_task = fill(NaN, L), fill(NaN, L), fill(NaN, q, L)
  for j in 1:L
    ok = findall(>=(j), Lk)
    isempty(ok) && continue
    w = wk[ok]
    cvm[j] = sum(w .* errs[ok, j]) / sum(w)
    length(ok) > 1 && (cvse[j] = sqrt(sum(w .* abs2.(errs[ok, j] .- cvm[j])) / sum(w) / (length(ok) - 1)))
    cvm_task[:, j] = [sum(sse[t, ok, j]) / sum(mt[t, ok]) for t in 1:q]
  end
  cvm, cvse, cvm_task, Lk
end

# λmin and λ1se of the summaries cvm and cvse: the level that minimizes cvm
# among those with a finite cvm, and the first level within cvse of it
function cv_levels(λs, cvm, cvse)
  ok = findall(!isnan, cvm)
  jmin = ok[argmin(cvm[ok])]
  bound = cvm[jmin] + (isnan(cvse[jmin]) ? 0.0 : cvse[jmin])
  λs[jmin], λs[findfirst(j -> cvm[j] <= bound, 1:jmin)]
end

# a ≈ b at rtol, with NaN in the same places
nan_approx(a, b; rtol = 1e-6) = isnan.(a) == isnan.(b) && isapprox(filter(!isnan, a), filter(!isnan, b); rtol)

@testset "multitask_lasso, MultitaskLassoPath and CVMultitaskLassoPath" begin

  @testset "one task: the lasso" begin
    rng = Xoshiro(81)
    Xs, ys = make_tasks(rng, (60,), 10)
    X, y = Xs[1], ys[1]
    λ = 0.3
    for intercept in (true, false), standardize_x in (true, false)
      sol = multitask_lasso(Xs, ys, λ, TIGHT; intercept, standardize_x)
      l = lasso(X, y, λ, TIGHT; intercept, standardize_x)
      @test Vector(sol.B[:, 1]) ≈ Vector(l.x) atol = 1e-6
      @test sol.intercept[1] ≈ l.intercept atol = 1e-6
    end
    l = lasso(X, y, λ, TIGHT; intercept = true, standardize_x = true)
    @test 0 < nnz(l.x) < 10
    # a task weight c: the lasso at λ/c; with one task :observations is c = 1
    sol = multitask_lasso(Xs, ys, λ, TIGHT; task_weights = [2.5])
    lc = lasso(X, y, λ / 2.5, TIGHT; intercept = true, standardize_x = true)
    @test Vector(sol.B[:, 1]) ≈ Vector(lc.x) atol = 1e-6
    @test sol.intercept[1] ≈ lc.intercept atol = 1e-6
    so = multitask_lasso(Xs, ys, λ, TIGHT; task_weights = :observations)
    @test Vector(so.B[:, 1]) ≈ Vector(l.x) atol = 1e-6
    @test so.intercept[1] ≈ l.intercept atol = 1e-6
  end

  @testset "a shared design: the multi-response lasso" begin
    rng = Xoshiro(82)
    n, p, q = 100, 15, 3
    X = centered_rand(rng, n, p) .* (1 .+ 2 .* rand(rng, p))'
    Bt = zeros(p, q); Bt[[1, 4, 7], :] .= centered_rand(rng, 3, q)
    Y = 1.0 .+ X * Bt .+ centered_rand(rng, n, q)
    ys = [Y[:, t] for t in 1:q]
    # the shared-design model (the same matrix, or equal copies), and the
    # stacked design, forced by the internal entry point or by a sparse copy
    cases = ((fill(X, q), true), ([copy(X) for _ in 1:q], true), (fill(X, q), false), (fill(sparse(X), q), true))
    fitwith(Xs, shared, λ; kw...) = shared ? multitask_lasso(Xs, ys, λ, TIGHT; kw...) :
      HighDimensionalModels._multitask_fit(Xs, ys, λ, TIGHT; shared, kw...)
    # the shared-design model (no stacked design Z) is taken for the same dense
    # matrix or equal copies with equal weights, and the stacked design otherwise
    data(Xs, tw = :equal; shared = true) =
      HighDimensionalModels._MultitaskData(Xs, ys; intercept = true, standardize_x = true, task_weights = tw, shared)
    @test data(fill(X, q)).Z === nothing && data([copy(X) for _ in 1:q]).Z === nothing
    @test data(fill(X, q), fill(2.0, q)).Z === nothing
    @test data(fill(sparse(X), q)).Z !== nothing && data(fill(X, q), [1.0, 2.0, 3.0]).Z !== nothing
    @test data(fill(X, q); shared = false).Z !== nothing
    for (intercept, standardize_x) in ((true, true), (false, false))
      λ = 0.3 * lambda_max(fill(X, q), ys, ones(q), ones(p), scales(fill(X, q), intercept, standardize_x); intercept)
      ms = multi_lasso(X, Y, λ, TIGHT; intercept, standardize_x)
      @test 0 < length(ms.activerows) < p
      for (Xs, shared) in cases
        sol = fitwith(Xs, shared, λ; intercept, standardize_x)
        @test Matrix(sol.B) ≈ Matrix(ms.B) atol = 1e-6
        @test sol.intercept ≈ ms.intercept atol = 1e-6
        @test sol.activerows == ms.activerows
        @test sol.nobs == q * n && sol.task_nobs == fill(n, q) && ms.nobs == n
      end
    end
    # a common task weight c: the multi-response lasso at λ/c; with equal
    # sizes :observations is :equal
    λ = 0.3 * lambda_max(fill(X, q), ys, ones(q), ones(p), scales(fill(X, q), true, true))
    ms = multi_lasso(X, Y, λ, TIGHT)
    for (Xs, shared) in cases
      sol = fitwith(Xs, shared, 2λ; task_weights = fill(2.0, q))
      @test Matrix(sol.B) ≈ Matrix(ms.B) atol = 1e-6
      @test sol.intercept ≈ ms.intercept atol = 1e-6
      so = fitwith(Xs, shared, λ; task_weights = :observations)
      @test so.task_weights == ones(q)
      @test Matrix(so.B) ≈ Matrix(ms.B) atol = 1e-6
    end
    # paths: the levels and solutions of MultiLassoPath, with the tolerance of
    # TIGHT (the default gap of 1e-4 stops each level about that far from the
    # solution)
    mp = MultiLassoPath(X, Y, TIGHT; nλ = 10)
    for (Xs, shared) in cases
      path = shared ? MultitaskLassoPath(Xs, ys, TIGHT; nλ = 10) :
        HighDimensionalModels._multitask_path(Xs, ys, nothing, TIGHT; nλ = 10, shared)
      @test path.λpath ≈ mp.λpath rtol = 1e-10
      @test all(norm(path.Bpath[k] - mp.Bpath[k]) < 1e-5 for k in eachindex(mp.λpath))
      @test all(norm(path.β0path[k] - mp.β0path[k]) < 1e-5 for k in eachindex(mp.λpath))
    end
  end

  @testset "unequal sizes: an independent reference" begin
    rng = Xoshiro(83)
    ns, p = (40, 80, 150), 12
    q = length(ns)
    Xs, ys = make_tasks(rng, ns, p)
    ω = 0.5 .+ rand(rng, p)
    for (intercept, standardize_x) in ((true, true), (false, true), (true, false), (false, false))
      S = scales(Xs, intercept, standardize_x)
      for (tw, pf) in ((:equal, nothing), (:observations, nothing), ([0.5, 2.0, 1.0], ω))
        c = weight_values(tw, ns)
        ωt = pf === nothing ? ones(p) : pf
        λ = 0.1 * lambda_max(Xs, ys, c, ωt, S; intercept)
        sol = multitask_lasso(Xs, ys, λ, TIGHT; intercept, standardize_x, task_weights = tw, penalty_factors = pf)
        ref = reference_fit(Xs, ys, λ, c, ωt, S; intercept)
        @test ref.converged
        @test kkt(Xs, ys, ref.β0, ref.B, λ, c, ωt, S; intercept) < 1e-8
        @test 0 < length(sol.activerows) < p
        @test sol.converged
        @test Matrix(sol.B) ≈ ref.B atol = 1e-6
        @test sol.intercept ≈ ref.β0 atol = 1e-6
        @test intercept || sol.intercept == zeros(q)
        @test objective(Xs, ys, sol.intercept, sol.B, λ, c, ωt, S) ≈
              objective(Xs, ys, ref.β0, ref.B, λ, c, ωt, S) atol = 1e-6
        # the optimality conditions of the documented objective
        @test kkt(Xs, ys, sol.intercept, sol.B, λ, c, ωt, S; intercept) < 1e-6
        # the scales, weights, residuals and objective of the data layer
        d = HighDimensionalModels._MultitaskData(Xs, ys; intercept, standardize_x, task_weights = tw)
        @test d.S ≈ S && d.c ≈ c
        @test HighDimensionalModels._task_residuals(d, sol.B, sol.intercept) ≈
              [ys[t] .- sol.intercept[t] .- Xs[t] * sol.B[:, t] for t in 1:q]
        @test HighDimensionalModels._task_objective(d, sol.B, sol.intercept, λ, ωt) ≈
              objective(Xs, ys, sol.intercept, sol.B, λ, c, ωt, S)
      end
    end
  end

  @testset "orthonormal designs: group soft-thresholding" begin
    # With centered columns X̃ₜ = Xₜ - 1x̄ₜ' such that X̃ₜ'X̃ₜ/nₜ = I, every
    # sₜⱼ = 1 and ‖ỹₜ - X̃ₜbₜ‖²/(2nₜ) = ‖bₜ - b̂ₜ‖²/2 + const, where
    # b̂ₜ = X̃ₜ'ỹₜ/nₜ is the least-squares coefficient of task t and ỹₜ the
    # centered response. The objective separates over the rows into
    # ∑ₜ cₜ (bₜⱼ - b̂ₜⱼ)²/2 + λωⱼ‖bⱼ‖, whose minimizer is zero when
    # ‖c ∘ b̂ⱼ‖ ≤ λωⱼ and otherwise has bₜⱼ = cₜ b̂ₜⱼ / (cₜ + λωⱼ/‖bⱼ‖)
    # (weighted_threshold). With cₜ = 1, ‖bⱼ‖ = ‖b̂ⱼ‖ - λωⱼ, which is group
    # soft-thresholding, bⱼ = (1 - λωⱼ/‖b̂ⱼ‖)₊ b̂ⱼ. The intercepts are
    # ȳₜ - x̄ₜ'bₜ.
    rng = Xoshiro(84)
    ns, p = (40, 80, 150), 6
    q = length(ns)
    Xs = map(collect(ns)) do n
      Z = centered_rand(rng, n, p)
      Z .-= mean(Z; dims = 1)
      sqrt(n) .* Matrix(qr(Z).Q) .+ centered_rand(rng, p)'
    end
    Bt = zeros(p, q); Bt[1:3, :] .= 1 .+ rand(rng, 3, q)
    ys = [2.0 .+ Xs[t] * Bt[:, t] .+ centered_rand(rng, ns[t]) for t in 1:q]
    @test maximum(abs, scales(Xs, true, true) .- 1) < 1e-12
    B̂ = reduce(hcat, [(X .- mean(X; dims = 1)) \ (y .- mean(y)) for (X, y) in zip(Xs, ys)])
    ω = [1.0, 0.5, 2.0, 1.0, 0.0, 1.5]
    λ = 0.3
    b = centered_rand(rng, q)
    @test weighted_threshold(b, ones(q), 0.4) ≈ max(1 - 0.4 / norm(b), 0) .* b rtol = 1e-14
    for tw in (:equal, :observations, [0.5, 2.0, 1.0])
      c = weight_values(tw, ns)
      Bc = tw === :equal ? reduce(vcat, [max(1 - λ * ω[j] / norm(B̂[j, :]), 0) .* B̂[j, :]' for j in 1:p]) :
                           reduce(vcat, [weighted_threshold(B̂[j, :], c, λ * ω[j])' for j in 1:p])
      β0 = [mean(ys[t]) - dot(vec(mean(Xs[t]; dims = 1)), Bc[:, t]) for t in 1:q]
      @test any(iszero, eachrow(Bc)) && any(!iszero, eachrow(Bc))
      sol = multitask_lasso(Xs, ys, λ, EXACT; task_weights = tw, penalty_factors = ω)
      @test sol.converged
      @test Matrix(sol.B) ≈ Bc atol = 1e-10
      @test sol.intercept ≈ β0 atol = 1e-10
      @test sol.activerows == [j for j in 1:p if any(!iszero, Bc[j, :])]
    end
  end

  @testset "special columns, rank deficiency, sparse and integer data, penalty factors" begin
    rng = Xoshiro(85)
    ns, p = (30, 50, 70), 8
    q = length(ns)

    # a column zero in every task, one constant within task 1 and one zero
    # within task 2: their coefficients there are zero, and the others fitted
    Xs, ys = make_tasks(rng, ns, p; support = [1, 3, 5, 7])
    for X in Xs
      X[:, 2] .= 0
    end
    Xs[1][:, 3] .= 2.5
    Xs[2][:, 5] .= 0
    for (intercept, standardize_x) in ((true, true), (true, false), (false, true))
      S = scales(Xs, intercept, standardize_x)
      @test S[2, :] == zeros(q) && S[5, 2] == 0 && (S[3, 1] == 0) == intercept
      d = HighDimensionalModels._MultitaskData(Xs, ys; intercept, standardize_x, task_weights = :equal)
      @test (d.S .== 0) == (S .== 0) && d.S ≈ S
      λ = 0.05 * lambda_max(Xs, ys, ones(q), ones(p), S; intercept)
      sol = multitask_lasso(Xs, ys, λ, TIGHT; intercept, standardize_x)
      @test all(isfinite, sol.B) && all(isfinite, sol.intercept)
      @test iszero(sol.B[2, :]) && iszero(sol.B[5, 2])
      @test iszero(sol.B[3, 1]) == intercept       # without an intercept, a nonzero constant is a predictor
      @test all(!iszero, sol.B[3, 2:3]) && all(!iszero, sol.B[5, [1, 3]])
      ref = reference_fit(Xs, ys, λ, ones(q), ones(p), S; intercept)
      @test ref.converged
      @test Matrix(sol.B) ≈ ref.B atol = 1e-6
      @test sol.intercept ≈ ref.β0 atol = 1e-6
      @test kkt(Xs, ys, sol.intercept, sol.B, λ, ones(q), ones(p), S; intercept) < 1e-6
    end

    # duplicated columns: the coefficients are not unique, but the problem is
    # the one without the copy (the penalty of two copies of a row is at least
    # that of their sum), with the same objective and fits
    Xs, ys = make_tasks(rng, ns, p)
    Xd = [[X X[:, 1]] for X in Xs]
    S, Sd = scales(Xs, true, true), scales(Xd, true, true)
    λ = 0.05 * lambda_max(Xs, ys, ones(q), ones(p), S)
    sol = multitask_lasso(Xs, ys, λ, TIGHT)
    for Xdd in (Xd, sparse.(Xd))
      sold = multitask_lasso(Xdd, ys, λ, TIGHT)
      @test objective(Xd, ys, sold.intercept, sold.B, λ, ones(q), ones(p + 1), Sd) ≈
            objective(Xs, ys, sol.intercept, sol.B, λ, ones(q), ones(p), S) atol = 1e-6
      @test maximum(maximum.(abs, predict(sold, Xd) .- predict(sol, Xs))) < 1e-6
      @test kkt(Xd, ys, sold.intercept, sold.B, λ, ones(q), ones(p + 1), Sd) < 1e-6
    end

    # fewer observations than predictors in two tasks, dense and sparse
    nr, pr = (8, 12, 40), 15
    Xr, yr = make_tasks(rng, nr, pr; support = 1:3)
    Sr = scales(Xr, true, true)
    λ = 0.1 * lambda_max(Xr, yr, ones(q), ones(pr), Sr)
    sr = multitask_lasso(Xr, yr, λ, TIGHT)
    srs = multitask_lasso(sparse.(Xr), yr, λ, TIGHT)
    @test kkt(Xr, yr, sr.intercept, sr.B, λ, ones(q), ones(pr), Sr) < 1e-6
    @test kkt(Xr, yr, srs.intercept, srs.B, λ, ones(q), ones(pr), Sr) < 1e-6
    ref = reference_fit(Xr, yr, λ, ones(q), ones(pr), Sr)
    @test ref.converged
    for s in (sr, srs)
      @test objective(Xr, yr, s.intercept, s.B, λ, ones(q), ones(pr), Sr) ≈
            objective(Xr, yr, ref.β0, ref.B, λ, ones(q), ones(pr), Sr) atol = 1e-6
      @test maximum(maximum.(abs, predict(s, Xr) .- [ref.β0[t] .+ Xr[t] * ref.B[:, t] for t in 1:q])) < 1e-6
    end

    # sparse designs against dense ones, with and without an intercept, on a
    # strongly convex problem; the prediction of sparse new designs
    Xsp = [sparse(centered_rand(rng, n, p) .* (rand(rng, n, p) .< 0.4)) for n in ns]
    ysp = [t .+ Xsp[t] * [1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0] .+ 0.5 .* centered_rand(rng, ns[t]) for t in 1:q]
    for intercept in (true, false), tw in (:equal, :observations)
      sd = multitask_lasso(Matrix.(Xsp), ysp, 0.05, TIGHT; intercept, task_weights = tw)
      ss = multitask_lasso(Xsp, ysp, 0.05, TIGHT; intercept, task_weights = tw)
      @test Matrix(ss.B) ≈ Matrix(sd.B) atol = 1e-6
      @test ss.intercept ≈ sd.intercept atol = 1e-6
      @test all(predict(ss, Xsp) .≈ predict(sd, Matrix.(Xsp)))
    end

    # integer data, a matrix that is not a Matrix, and tasks of different
    # element types
    Xi = [floor.(Int, 7 .* rand(rng, n, p)) .- 3 for n in ns]
    yi = [X * [2, -1, 1, 0, 0, 0, 0, 0] .+ floor.(Int, 5 .* rand(rng, size(X, 1))) .- 2 for X in Xi]
    si = multitask_lasso(Xi, yi, 1, TIGHT)
    sf = multitask_lasso(float.(Xi), float.(yi), 1.0, TIGHT)
    @test si.B isa SparseMatrixCSC{Float64, Int} && si.intercept isa Vector{Float64}
    @test 0 < length(si.activerows) < p
    @test Matrix(si.B) ≈ Matrix(sf.B) atol = 1e-6
    @test MultitaskLassoPath(Xi, yi; nλ = 5).λpath isa Vector{Float64}
    sm = multitask_lasso([permutedims(Xi[1])', Float64.(Xi[2]), Xi[3]], yi, 1, TIGHT)
    @test sm.B isa SparseMatrixCSC{Float64, Int}
    @test Matrix(sm.B) ≈ Matrix(sf.B) atol = 1e-6
    s32 = multitask_lasso([Xi[1], Float32.(Xi[2]), Xi[3]], yi, 1)
    @test s32.B isa SparseMatrixCSC{Float32, Int} && s32.intercept isa Vector{Float32}
    @test Matrix(s32.B) ≈ Matrix(sf.B) atol = 1e-4
    # Float32 paths: with the default gap rule, ProximalSolvers 0.2.0 has no
    # method of its gap check for the stacked model in Float32 (a MethodError);
    # with gaptol = 0 they run. The Float32 grid differs from the Float64 one
    # by the factor 1 + √eps(T) of λmax, so the Float64 path is fitted on it.
    X32, y32 = [Float32.(X) for X in Xi], [Float32.(y) for y in yi]
    @test_broken (MultitaskLassoPath(X32, y32; nλ = 10); true)
    @test_broken (CVMultitaskLassoPath(X32, y32; nfolds = 3, nλ = 10); true)
    p32 = MultitaskLassoPath(X32, y32, CDOptions(; gaptol = 0); nλ = 10)
    @test p32.λpath isa Vector{Float32} && p32.Bpath isa Vector{SparseMatrixCSC{Float32, Int}} && all(p32.converged)
    λs64 = MultitaskLassoPath(Xi, yi; nλ = 10).λpath
    @test p32.λpath ./ (1 + sqrt(eps(Float32))) ≈ λs64 ./ (1 + sqrt(eps())) rtol = 1e-5
    p64 = MultitaskLassoPath(Xi, yi, Float64.(p32.λpath), TIGHT)
    @test all(norm(p32.Bpath[k] - p64.Bpath[k]) < 1e-4 for k in 1:10)
    cv32 = CVMultitaskLassoPath(X32, y32, CDOptions(; gaptol = 0); nfolds = 3, nλ = 10, rng = Xoshiro(2))
    @test cv32.cvm isa Vector{Float32} && all(isfinite, cv32.cvm)

    # a zero penalty factor: allowed with an explicit level or grid, where the
    # predictor stays in the model, and rejected with an automatic grid
    Xs, ys = make_tasks(rng, ns, p)
    S = scales(Xs, true, true)
    ω = ones(p); ω[6] = 0
    λmax = lambda_max(Xs, ys, ones(q), ones(p), S)
    sol = multitask_lasso(Xs, ys, 0.3λmax, TIGHT; penalty_factors = ω)
    ref = reference_fit(Xs, ys, 0.3λmax, ones(q), ω, S)
    @test 6 in sol.activerows && sol.penalty_factors == ω
    @test Matrix(sol.B) ≈ ref.B atol = 1e-6
    λs = λmax .* [1.0, 0.5, 0.2, 0.05]
    path = MultitaskLassoPath(Xs, ys, λs, TIGHT; penalty_factors = ω)
    @test path.λpath == λs && path.penalty_factors == ω
    @test all(6 in a for a in path.activerows)
    @test all(kkt(Xs, ys, path.β0path[k], path.Bpath[k], λs[k], ones(q), ω, S) < 1e-6 for k in eachindex(λs))
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys; penalty_factors = ω)
    @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys; penalty_factors = ω, nfolds = 3)
  end

  @testset "paths" begin
    rng = Xoshiro(86)
    ns, p = (40, 80, 150), 12
    q, N = length(ns), sum(ns)
    Xs, ys = make_tasks(rng, ns, p)
    S = scales(Xs, true, true)
    for tw in (:equal, :observations)
      c = weight_values(tw, ns)
      λmax = lambda_max(Xs, ys, c, ones(p), S)
      path = MultitaskLassoPath(Xs, ys, TIGHT; nλ = 25, task_weights = tw)
      L = length(path.λpath)
      @test L == 25 && all(path.converged) && path.intercept
      @test length(path.Bpath) == length(path.β0path) == length(path.activerows) == length(path.iterations) ==
            length(path.kkt_checks) == L
      @test path.nobs == N && path.task_nobs == collect(ns) && path.task_weights ≈ c
      @test path.penalty_factors == ones(p)
      # λmax: the first level is zero, and a level slightly below λmax is not
      @test path.λpath[1] ≈ λmax * (1 + sqrt(eps())) rtol = 1e-8
      @test path.λpath[end] / path.λpath[1] ≈ 1e-4 rtol = 1e-8
      @test nnz(path.Bpath[1]) == 0 && isempty(path.activerows[1]) && !isempty(path.activerows[2])
      @test path.β0path[1] ≈ mean.(ys) atol = 1e-10
      @test nnz(multitask_lasso(Xs, ys, λmax * (1 - 1e-3); task_weights = tw).B) > 0
      @test nnz(multitask_lasso(Xs, ys, λmax * (1 + 1e-6); task_weights = tw).B) == 0
      # the levels are the single fits, with the tolerance of TIGHT (the
      # default gap of 1e-4 stops each level about that far from the solution)
      for k in (6, 25)
        s = multitask_lasso(Xs, ys, path.λpath[k], TIGHT; task_weights = tw)
        @test Matrix(path.Bpath[k]) ≈ Matrix(s.B) atol = 1e-5
        @test path.β0path[k] ≈ s.intercept atol = 1e-5
        @test path.activerows[k] == [j for j in 1:p if any(!iszero, path.Bpath[k][j, :])]
      end
      # screening or not
      path0 = MultitaskLassoPath(Xs, ys, path.λpath, TIGHT; task_weights = tw, screening = false)
      @test all(norm(path.Bpath[k] - path0.Bpath[k]) < 1e-5 for k in 1:L)
      @test all(norm(path.β0path[k] - path0.β0path[k]) < 1e-5 for k in 1:L)
    end
    # the path stops after the first level with more than max_rows rows
    pathm = MultitaskLassoPath(Xs, ys; nλ = 50, max_rows = 4)
    @test length(pathm.λpath) < 50 && length(pathm.Bpath) == length(pathm.λpath)
    @test length(last(pathm.activerows)) > 4 && all(length(a) <= 4 for a in pathm.activerows[1:end-1])
    # coef along the path and interpolated in λ
    path = MultitaskLassoPath(Xs, ys; nλ = 10)
    C = coef(path)
    @test length(C) == 10 && all(size(Ck) == (p + 1, q) for Ck in C)
    @test all(C[k] ≈ [path.β0path[k]'; Matrix(path.Bpath[k])] for k in 1:10)
    @test coef(path, path.λpath[4]) ≈ C[4]
    λ = 0.3 * path.λpath[3] + 0.7 * path.λpath[4]
    t = (path.λpath[3] - λ) / (path.λpath[3] - path.λpath[4])
    @test Matrix(coef(path, λ)) ≈ (1 - t) .* Matrix(C[3]) .+ t .* Matrix(C[4])
    @test_throws ArgumentError coef(path, 2 * path.λpath[1])
    @test_throws ArgumentError coef(path, path.λpath[end] / 2)
    # the default λminratio with fewer observations than coefficients, and a
    # range of levels
    Xr, yr = make_tasks(rng, (10, 15), 20; support = 1:2)
    pr = MultitaskLassoPath(Xr, yr; nλ = 8)
    @test pr.λpath[end] / pr.λpath[1] ≈ 0.01 rtol = 1e-8
    pg = MultitaskLassoPath(Xs, ys, range(path.λpath[1], path.λpath[5]; length = 4))
    @test length(pg.λpath) == 4 && pg.λpath[4] ≈ path.λpath[5]
    # one task: the levels of LassoPath
    lp = LassoPath(Xs[1], ys[1]; nλ = 10, intercept = true, standardize_x = true)
    @test MultitaskLassoPath(Xs[1:1], ys[1:1]; nλ = 10).λpath ≈ lp.λpath rtol = 1e-10
  end

  @testset "cross-validation" begin
    rng = Xoshiro(87)
    ns, p = (43, 81, 150), 12
    q = length(ns)
    Xs, ys = make_tasks(rng, ns, p)
    # random folds, balanced within each task, reproducible with or without
    # threads
    cv1 = CVMultitaskLassoPath(Xs, ys; nfolds = 4, nλ = 10, rng = Xoshiro(3))
    cv2 = CVMultitaskLassoPath(Xs, ys; nfolds = 4, nλ = 10, rng = Xoshiro(3), threads = false)
    @test cv1.folds == cv2.folds && cv1.cvm == cv2.cvm && cv1.cvse == cv2.cvse && cv1.cvm_task == cv2.cvm_task
    @test cv1.folds isa Vector{Vector{Int}} && length.(cv1.folds) == collect(ns)
    for (t, n) in enumerate(ns)
      counts = [count(==(k), cv1.folds[t]) for k in 1:4]
      @test sum(counts) == n && all(in((fld(n, 4), cld(n, 4))), counts)
    end
    # the errors of the folds, recomputed from paths fitted on the training
    # rows alone, with the full data's task weights
    folds = [repeat(1:5; outer = cld(n, 5))[1:n] for n in ns]
    for tw in (:equal, :observations)
      c = weight_values(tw, ns)
      cv = CVMultitaskLassoPath(Xs, ys, TIGHT; folds, nλ = 20, task_weights = tw, rng = Xoshiro(5))
      L = length(cv.λpath)
      @test cv.folds == folds && cv.λpath == cv.path.λpath && L == 20
      @test length(cv.cvm) == length(cv.cvse) == length(cv.nrows) == L && size(cv.cvm_task) == (q, L)
      @test cv.nrows == length.(cv.path.activerows)
      @test cv.task_weights ≈ c && cv.path.task_weights ≈ c
      full = MultitaskLassoPath(Xs, ys, TIGHT; nλ = 20, task_weights = tw)
      @test cv.λpath ≈ full.λpath rtol = 1e-12
      @test all(norm(cv.path.Bpath[k] - full.Bpath[k]) < 1e-5 for k in 1:L)
      cvm, cvse, cvm_task, Lk = cv_recompute(cv, Xs, ys, c, tw)
      @test Lk == fill(L, 5)
      @test cv.cvm ≈ cvm rtol = 1e-6
      @test cv.cvse ≈ cvse rtol = 1e-6
      @test cv.cvm_task ≈ cvm_task rtol = 1e-6
      jmin = argmin(cv.cvm)
      @test cv.λmin == cv.λpath[jmin]
      @test cv.λ1se == cv.λpath[findfirst(j -> cv.cvm[j] <= cv.cvm[jmin] + cv.cvse[jmin], 1:jmin)]
    end
    # fold paths that stop early (max_rows): a level is averaged over the folds
    # that reached it. The data are the first of a range of seeds for which
    # some fold paths stop before the last level and the others reach it.
    partial = nothing
    for seed in 1:100
      Xa, ya = make_tasks(Xoshiro(seed), ns, p)
      cv = CVMultitaskLassoPath(Xa, ya, TIGHT; folds, nλ = 20, max_rows = 4, rng = Xoshiro(7))
      r = cv_recompute(cv, Xa, ya, ones(q), :equal; max_rows = 4)
      L, Lk = length(cv.λpath), r[4]
      L < 20 && minimum(Lk) < L == maximum(Lk) && (partial = (cv, r); break)
    end
    @test partial !== nothing
    if partial !== nothing
      cv, (cvm, cvse, cvm_task, Lk) = partial
      L = length(cv.λpath)
      @test L < 20 && minimum(Lk) < L == maximum(Lk)
      @test all(isfinite, cv.cvm) && all(isfinite, cv.cvm_task)
      @test nan_approx(cv.cvm, cvm)
      @test nan_approx(cv.cvse, cvse)
      @test nan_approx(cv.cvm_task, cvm_task)
      @test (cv.λmin, cv.λ1se) == cv_levels(cv.λpath, cv.cvm, cv.cvse)
    end
    # no fold path reaches the last levels of the full path: NaN there, the
    # same lengths, and λmin and λ1se among the levels reached. The data are
    # the first of a range of seeds for which no fold path reaches the last
    # level, with either option set.
    nb, pb = (20, 30, 40), 40
    fb = [repeat(1:3; outer = cld(n, 3))[1:n] for n in nb]
    unreached = nothing
    for seed in 1:100
      rng = Xoshiro(seed)
      Bb = zeros(pb, 3); Bb[1:15, :] .= centered_rand(rng, 15, 3)
      Xb = [centered_rand(rng, n, pb) for n in nb]
      yb = [Xb[t] * Bb[:, t] .+ centered_rand(rng, nb[t]) for t in 1:3]
      runs = map((CDOptions(), TIGHT)) do options
        cv = CVMultitaskLassoPath(Xb, yb, options; folds = fb, nλ = 40, max_rows = 6, rng = Xoshiro(1))
        options, cv, cv_recompute(cv, Xb, yb, ones(3), :equal, options; max_rows = 6)
      end
      all(((_, cv, r),) -> maximum(r[4]) < length(cv.λpath), runs) && (unreached = runs; break)
    end
    @test unreached !== nothing
    for (options, cv, (cvm, cvse, cvm_task, Lk)) in something(unreached, ())
      L = length(cv.λpath)
      reached = maximum(Lk)
      @test reached < L
      @test length(cv.cvm) == length(cv.cvse) == length(cv.nrows) == L && size(cv.cvm_task) == (3, L)
      @test all(isnan, cv.cvm[reached+1:L]) && all(isnan, cv.cvse[reached+1:L])
      @test all(isnan, cv.cvm_task[:, reached+1:L])
      @test all(isfinite, cv.cvm[1:reached]) && all(isfinite, cv.cvm_task[:, 1:reached])
      @test cv.λmin in cv.λpath[1:reached] && cv.λ1se in cv.λpath[1:reached]
      @test (cv.λmin, cv.λ1se) == cv_levels(cv.λpath, cv.cvm, cv.cvse)
      if options === TIGHT
        @test nan_approx(cv.cvm, cvm)
        @test nan_approx(cv.cvse, cvse)
        @test nan_approx(cv.cvm_task, cvm_task)
      end
    end
    # one task with the same folds: CVLassoPath
    cvl = CVLassoPath(Xs[3], ys[3], TIGHT; folds = folds[3], nλ = 20, intercept = true, standardize_x = true,
                      rng = Xoshiro(6))
    cv = CVMultitaskLassoPath(Xs[3:3], ys[3:3], TIGHT; folds = folds[3:3], nλ = 20, rng = Xoshiro(6))
    @test cv.λpath ≈ cvl.λpath rtol = 1e-10
    @test cv.cvm ≈ cvl.cvm rtol = 1e-6
    @test cv.cvse ≈ cvl.cvse rtol = 1e-6
    @test cv.λmin ≈ cvl.λmin rtol = 1e-10
    @test cv.λ1se ≈ cvl.λ1se rtol = 1e-10
    @test vec(cv.cvm_task) ≈ cvl.cvm rtol = 1e-6
    # coef and predict at λ1se and λmin
    cv = CVMultitaskLassoPath(Xs, ys; folds, nλ = 15)
    @test coef(cv) == coef(cv.path, cv.λ1se)
    @test coef(cv; λ = :λmin) == coef(cv.path, cv.λmin)
    C = Matrix(coef(cv; λ = :λmin))
    P = predict(cv, Xs; λ = :λmin)
    @test length(P) == q && all(P[t] ≈ C[1, t] .+ Xs[t] * C[2:end, t] for t in 1:q)
    j = findfirst(==(cv.λ1se), cv.λpath)
    Pp = predict(cv.path, Xs)
    @test all(predict(cv, Xs)[t] ≈ Pp[t][:, j] for t in 1:q)
    @test_throws ArgumentError coef(cv; λ = :bogus)
    @test fit(CVMultitaskLassoPath, Xs, ys; folds, nλ = 15).λpath == cv.λpath
    # the validation of the folds
    bad = deepcopy(folds); bad[2][bad[2] .== 5] .= 4                 # label 5 unused in task 2
    @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys; folds = bad, nλ = 5)
    @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys; folds = [ones(Int, n) for n in ns], nλ = 5)
    @test_throws DimensionMismatch CVMultitaskLassoPath(Xs, ys; folds = folds[1:2], nλ = 5)
    @test_throws DimensionMismatch CVMultitaskLassoPath(Xs, ys; folds = [folds[1][1:end-1], folds[2], folds[3]],
                                                         nλ = 5)
    @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys; nfolds = 1, nλ = 5)
    # a task too small for the folds: fewer observations than folds, or a
    # training fold of one observation with an intercept
    Xt = [Xs[1], Xs[2][1:4, :], Xs[3]]
    yt = [ys[1], ys[2][1:4], ys[3]]
    @test_throws ArgumentError CVMultitaskLassoPath(Xt, yt; nfolds = 5, nλ = 5)
    Xt = [Xs[1], Xs[2][1:2, :], Xs[3]]
    yt = [ys[1], ys[2][1:2], ys[3]]
    ft = [repeat(1:2; outer = cld(n, 2))[1:n] for n in length.(yt)]
    @test_throws ArgumentError CVMultitaskLassoPath(Xt, yt; folds = ft, nλ = 5)
    @test CVMultitaskLassoPath(Xt, yt; folds = ft, nλ = 5, intercept = false).folds == ft
  end

  @testset "invariances" begin
    rng = Xoshiro(88)
    ns, p = (40, 80, 150), 12
    q = length(ns)
    Xs, ys = make_tasks(rng, ns, p)
    λ = 0.2
    for tw in (:equal, [0.5, 2.0, 1.0])
      sol = multitask_lasso(Xs, ys, λ, TIGHT; task_weights = tw)
      B = Matrix(sol.B)
      @test 0 < length(sol.activerows) < p && B[3, 2] != 0 && B[4, 1] != 0
      # rescaling a column within a task rescales its coefficient there, and
      # shifting one changes the intercept only
      Xa = deepcopy(Xs)
      Xa[2][:, 3] .*= 5
      Xa[1][:, 4] .+= 3
      sa = multitask_lasso(Xa, ys, λ, TIGHT; task_weights = tw)
      Ba = copy(B); Ba[3, 2] /= 5
      @test Matrix(sa.B) ≈ Ba atol = 1e-6
      @test sa.intercept ≈ sol.intercept .- [3 * B[4, 1], 0, 0] atol = 1e-6
      @test all(predict(sa, Xa) .≈ predict(sol, Xs))
      # permuting the tasks permutes the columns
      perm = [3, 1, 2]
      sp = multitask_lasso(Xs[perm], ys[perm], λ, TIGHT; task_weights = tw isa Symbol ? tw : tw[perm])
      @test Matrix(sp.B) ≈ B[:, perm] atol = 1e-6
      @test sp.intercept ≈ sol.intercept[perm] atol = 1e-6
    end
    sp = multitask_lasso(Xs[[3, 1, 2]], ys[[3, 1, 2]], λ, TIGHT; task_weights = :observations)
    so = multitask_lasso(Xs, ys, λ, TIGHT; task_weights = :observations)
    @test Matrix(sp.B) ≈ Matrix(so.B)[:, [3, 1, 2]] atol = 1e-6
    # duplicating the rows of a task: :equal does not change, :observations
    # gives the task the weight of its new size
    Xd = [[Xs[1]; Xs[1]], Xs[2], Xs[3]]
    yd = [[ys[1]; ys[1]], ys[2], ys[3]]
    nd = [80, 80, 150]
    se, sed = multitask_lasso(Xs, ys, λ, TIGHT), multitask_lasso(Xd, yd, λ, TIGHT)
    @test Matrix(sed.B) ≈ Matrix(se.B) atol = 1e-6
    @test sed.intercept ≈ se.intercept atol = 1e-6
    sod = multitask_lasso(Xd, yd, λ, TIGHT; task_weights = :observations)
    @test sod.task_weights ≈ 3 .* nd ./ sum(nd) && so.task_weights ≈ 3 .* collect(ns) ./ sum(ns)
    @test norm(Matrix(sod.B) - Matrix(so.B)) > 1e-3
    sw = multitask_lasso(Xs, ys, λ, TIGHT; task_weights = 3 .* nd ./ sum(nd))
    @test Matrix(sod.B) ≈ Matrix(sw.B) atol = 1e-6
  end

  @testset "StatsAPI and the validation of the inputs" begin
    rng = Xoshiro(89)
    ns, p = (40, 80, 150), 12
    q, N = length(ns), sum(ns)
    Xs, ys = make_tasks(rng, ns, p)
    Xn = [centered_rand(rng, m, p) for m in (5, 7, 3)]
    λ = 0.2
    sol = multitask_lasso(Xs, ys, λ)
    @test sol isa MultitaskLassoSolution
    @test sol.B isa SparseMatrixCSC{Float64, Int} && size(sol.B) == (p, q) && length(sol.intercept) == q
    @test sol.λ == λ && sol.hasintercept && sol.converged && sol.iterations isa Int
    @test sol.nobs == N && sol.task_nobs == collect(ns) && sol.task_weights == ones(q)
    @test sol.penalty_factors == ones(p)
    @test sol.activerows == [j for j in 1:p if any(!iszero, sol.B[j, :])]
    @test coef(sol) == [sol.intercept'; Matrix(sol.B)] && size(coef(sol)) == (p + 1, q)
    P = predict(sol, Xn)
    @test length(P) == q && all(P[t] ≈ sol.intercept[t] .+ Xn[t] * sol.B[:, t] for t in 1:q)
    @test length.(P) == [5, 7, 3]
    @test nobs(sol) == N && dof(sol) == nnz(sol.B) + q
    s0 = multitask_lasso(Xs, ys, λ; intercept = false)
    @test !s0.hasintercept && s0.intercept == zeros(q)
    @test coef(s0) == s0.B && dof(s0) == nnz(s0.B)
    @test all(predict(s0, Xn) .≈ [Xn[t] * s0.B[:, t] for t in 1:q])
    @test_throws DimensionMismatch predict(sol, [X[:, 1:end-1] for X in Xn])
    @test_throws DimensionMismatch predict(sol, Xn[1:2])

    path = MultitaskLassoPath(Xs, ys; nλ = 8)
    @test path isa MultitaskLassoPath
    @test fit(MultitaskLassoPath, Xs, ys; nλ = 8).λpath == path.λpath
    @test fit(MultitaskLassoPath, Xs, ys, path.λpath[1:3]).λpath == path.λpath[1:3]
    @test nobs(path) == N
    Pp = predict(path, Xn)
    @test length(Pp) == q && all(size(Pp[t]) == (size(Xn[t], 1), 8) for t in 1:q)
    @test all(Pp[t][:, k] ≈ path.β0path[k][t] .+ Xn[t] * path.Bpath[k][:, t] for t in 1:q, k in 1:8)
    p0 = MultitaskLassoPath(Xs, ys; nλ = 4, intercept = false)
    @test !p0.intercept && all(coef(p0) .== p0.Bpath) && all(b -> b == zeros(q), p0.β0path)

    # the inputs
    @test_throws ArgumentError multitask_lasso(Matrix{Float64}[], Vector{Float64}[], λ)
    @test_throws DimensionMismatch multitask_lasso(Xs, ys[1:2], λ)
    @test_throws DimensionMismatch multitask_lasso([Xs[1], Xs[2][:, 1:end-1], Xs[3]], ys, λ)
    @test_throws DimensionMismatch multitask_lasso(Xs, [ys[1], ys[2][1:end-1], ys[3]], λ)
    @test_throws ArgumentError multitask_lasso([Xs[1][1:1, :], Xs[2]], [ys[1][1:1], ys[2]], λ)
    @test multitask_lasso([Xs[1][1:1, :], Xs[2]], [ys[1][1:1], ys[2]], λ; intercept = false).task_nobs == [1, 80]
    @test_throws ArgumentError multitask_lasso([Xs[1][1:0, :], Xs[2]], [ys[1][1:0], ys[2]], λ; intercept = false)
    Xbad = deepcopy(Xs); Xbad[2][3, 4] = NaN
    @test_throws ArgumentError multitask_lasso(Xbad, ys, λ)
    ybad = deepcopy(ys); ybad[3][7] = Inf
    @test_throws ArgumentError multitask_lasso(Xs, ybad, λ)
    @test_throws ArgumentError multitask_lasso(Xs, ys, -0.1)
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys, [0.2, -0.1])
    for tw in (:bogus, [1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, Inf, 1.0], [1.0, NaN, 1.0])
      @test_throws ArgumentError multitask_lasso(Xs, ys, λ; task_weights = tw)
    end
    @test_throws DimensionMismatch multitask_lasso(Xs, ys, λ; task_weights = [1.0, 1.0])
    for ω in ([-1.0; ones(p - 1)], zeros(p), [Inf; ones(p - 1)], [NaN; ones(p - 1)])
      @test_throws ArgumentError multitask_lasso(Xs, ys, λ; penalty_factors = ω)
    end
    @test_throws DimensionMismatch multitask_lasso(Xs, ys, λ; penalty_factors = ones(p - 1))
    # options are passed as a CDOptions, and other keywords are rejected
    @test_throws ArgumentError multitask_lasso(Xs, ys, λ; bogus = 1)
    @test_throws ArgumentError multitask_lasso(Xs, ys, λ; opt_tol = 1e-8)
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys; bogus = 1)
    @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys; bogus = 1)
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys; nλ = 0)
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys; λminratio = 0)
    @test_throws ArgumentError MultitaskLassoPath(Xs, ys; λminratio = 2)
  end
end

end
