module MultitaskDirtyTest

using HighDimensionalModels
using DelimitedFiles, LinearAlgebra, Random, SparseArrays, Statistics, Test

const HDM = HighDimensionalModels
const TIGHT = CDOptions(; opt_tol = 1e-13, gaptol = 1e-12, order = :cyclic)
const EXACT = CDOptions(TIGHT; opt_tol = 1e-14, gaptol = 0)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)
read_values(file) = Dict(row[1] => row[2] for row in eachrow(readdlm(file, ',', String)))
read_matrix(dir, file) = readdlm(joinpath(dir, file), ',', Float64)

# Scales computed from the original designs, independently of preprocessing.
function scales(Xs, intercept, standardize)
  hcat([map(axes(X, 2)) do j
          x = Vector(X[:, j])
          (intercept ? all(==(first(x)), x) : all(iszero, x)) && return 0.0
          standardize ? sqrt(mean(abs2, x .- (intercept ? mean(x) : 0))) : 1.0
        end for X in Xs]...)
end

# Dual feasibility and the two Fenchel equalities at the returned point.
function certificate(Xs, ys, sol, scales)
  B, A, S = Matrix(sol.B), Matrix(sol.A), Matrix(sol.S)
  U, V = scales .* A, scales .* S
  Z = zeros(size(B))
  err = maximum(abs, B - A - S; init = 0.0) / max(1, norm(B, Inf), norm(A, Inf), norm(S, Inf))
  loss = 0.0
  for t in eachindex(ys)
    r = ys[t] .- sol.intercept[t] .- Xs[t] * B[:, t]
    c, n = sol.task_weights[t], length(r)
    loss += c * sum(abs2, r) / (2n)
    if sol.hasintercept
      err = max(err, c * abs(mean(r)) / max(1, c * sqrt(mean(abs2, r))))
    end
    for j in axes(B, 1)
      if scales[j, t] > 0
        x = Xs[t][:, j] .- (sol.hasintercept ? mean(Xs[t][:, j]) : 0)
        Z[j, t] = c * dot(x, r) / (n * scales[j, t])
      else
        err = max(err, abs(B[j, t]), abs(A[j, t]), abs(S[j, t]))
      end
    end
  end
  penalty = 0.0
  for j in axes(B, 1)
    z, a, s = Z[j, :], U[j, :], V[j, :]
    cb, cs = sol.penalty_factors[j] * sol.λ_shared, sol.penalty_factors[j] * sol.λ_specific
    nz = sol.shared_norm === :linf ? norm(z, 1) : norm(z)
    na = sol.shared_norm === :linf ? norm(a, Inf) : norm(a)
    pb, ps = cb * na, cs * norm(s, 1)
    err = max(err, (nz - cb) / max(1, nz, cb), (norm(z, Inf) - cs) / max(1, norm(z, Inf), cs),
              abs(dot(z, a) - pb) / max(1, abs(dot(z, a)), pb),
              abs(dot(z, s) - ps) / max(1, abs(dot(z, s)), ps))
    penalty += pb + ps
  end
  (; kkt = err, objective = loss + penalty)
end

function check_solution(sol, Xs, ys, scales)
  cert = certificate(Xs, ys, sol, scales)
  @test sol.converged
  @test cert.kkt <= 1e-6
  @test sol.kkt ≈ cert.kkt atol = 2e-13
  @test sol.objective ≈ cert.objective rtol = 1e-13 atol = 1e-14
  @test Matrix(sol.B) ≈ Matrix(sol.A + sol.S) atol = 1e-14
  @test sol.activerows == findall(j -> any(!iszero, sol.B[j, :]), axes(sol.B, 1))
  @test sol.sharedrows == findall(j -> any(!iszero, sol.A[j, :]), axes(sol.A, 1))
  # An independent clipping rule also checks the rescaling of components.
  for j in axes(sol.B, 1)
    on = scales[j, :] .> 0
    v = scales[j, on] .* Vector(sol.B[j, on])
    a = scales[j, on] .* Vector(sol.A[j, on])
    r = iszero(sol.λ_specific) ? Inf : sol.λ_shared / sol.λ_specific
    endpoint = sol.shared_norm === :linf ? length(v) : sqrt(count(!iszero, v))
    expected = if iszero(sol.λ_shared)
      v
    elseif iszero(sol.λ_specific) || r >= endpoint
      zeros(length(v))
    elseif r <= 1
      v
    elseif sol.shared_norm === :linf
      threshold = sort(abs.(v); rev = true)[ceil(Int, r)]
      clamp.(v, -threshold, threshold)
    else
      # Bisection is independent of the implementation's sorted intervals.
      lo, hi = 0.0, norm(v) / r
      for _ in 1:80
        mid = (lo + hi) / 2
        if sum(min(abs(x) / mid, 1)^2 for x in v) > r^2
          lo = mid
        else
          hi = mid
        end
      end
      clamp.(v, -hi, hi)
    end
    @test a ≈ expected atol = 2e-12
  end
end

# The threshold at zero, computed directly from the centered data.
function lambda_max(Xs, ys, scales, c, ω, ratio, shared_norm; intercept = true)
  maximum(axes(scales, 1)) do j
    z = [iszero(scales[j, t]) ? 0.0 :
         c[t] * dot(Xs[t][:, j], ys[t] .- (intercept ? mean(ys[t]) : 0)) / (length(ys[t]) * scales[j, t])
         for t in eachindex(ys)]
    max((shared_norm === :linf ? norm(z, 1) : norm(z)) / ratio, norm(z, Inf)) / ω[j]
  end
end

@testset "Dirty multitask" begin
  @testset "Canonical row splits" begin
    v = [5.0, -3, 1, 1]
    for r in (1.5, 2.0, 2 - 1e-9)
      a, s = HDM._dirty_split(v, r, 1.0, :linf)
      @test a == [3, -3, 1, 1] && a + s == v
    end
    a, s = HDM._dirty_split(v, 2 + 1e-9, 1.0, :linf)
    @test a == [1, -1, 1, 1] && a + s == v
    a, _ = HDM._dirty_split([4.0, -2, 2, 0], 2.0, 1.0, :linf)
    @test a == [2, -2, 2, 0]
    v2, r = [3.0, -1, 0.5], 1.2
    a, s = HDM._dirty_split(v2, r, 1.0, :l2)
    τ = sqrt(1.25 / (r^2 - 1))
    @test a ≈ [τ, -1, 0.5] && a + s ≈ v2
    @test sum(min(abs(x) / τ, 1)^2 for x in v2) ≈ r^2
    @test HDM._dirty_split([2.0, -2, 2], 1.3, 1.0, :l2)[1] == [2, -2, 2]
    for shared_norm in (:linf, :l2)
      endpoint = shared_norm === :linf ? length(v) : sqrt(count(!iszero, v))
      for r in (endpoint, endpoint + 1)
        a, s = HDM._dirty_split(v, r, 1.0, shared_norm)
        @test all(iszero, a) && s == v
      end
      for r in (0.5, 1.0)
        a, s = HDM._dirty_split(v, r, 1.0, shared_norm)
        @test a == v && all(iszero, s)
      end
      for (lb, ls) in ((0.0, 1.0), (0.0, 0.0))
        a, s = HDM._dirty_split(v, lb, ls, shared_norm)
        @test a == v && all(iszero, s)
      end
      a, s = HDM._dirty_split(v, 1.0, 0.0, shared_norm)
      @test all(iszero, a) && s == v
      @test all(iszero, first(HDM._dirty_split(zeros(3), 1.2, 1.0, shared_norm)))
    end
    @test all(iszero, first(HDM._dirty_split([3.0, 0, 0], 1.0, 1.0, :l2)))
  end

  @testset "Row values, proximal steps and local buffers" begin
    Xs, ys = fill([1.0 0; -1 1; 0 -1], 4), fill([1.0, -1, 0], 4)
    d = HDM._MultitaskData(Xs, ys; intercept = false, standardize_x = false, task_weights = :equal, stacked = false)
    for shared_norm in (:linf, :l2), r in (0.6, 1.0, 1.3, 2.0, 2.0 + 1e-9, 5.0)
      pen = HDM._DirtyRowPenalty(d, [1.7, 0.0], r, shared_norm)
      other = HDM._DirtyRowPenalty(d, [1.7, 0.0], r, shared_norm)
      @test pen.slopes[1].vals !== other.slopes[1].vals
      for u in ([3.0, -1, 0.5, 0], [2.0, -2, 2, -2], zeros(4))
        step = 0.4
        v = HDM._row_prox!(copy(u), pen, 1, step, zeros(4))
        z = (u - v) / step
        a, s = HDM._dirty_split(v, r, 1.0, shared_norm)
        cb, cs = 1.7r, 1.7
        @test (shared_norm === :linf ? norm(z, 1) : norm(z)) <= cb + 1e-12
        @test norm(z, Inf) <= cs + 1e-12
        @test dot(z, a) ≈ cb * (shared_norm === :linf ? norm(a, Inf) : norm(a)) atol = 2e-12
        @test dot(z, s) ≈ cs * norm(s, 1) atol = 2e-12
        @test HDM._row_value(v, pen, 1) ≈ cb * (shared_norm === :linf ? norm(a, Inf) : norm(a)) + cs * norm(s, 1)
        @test HDM._row_prox!(copy(u), pen, 2, step, zeros(4)) == u
        @test HDM._row_dual(u, pen, 1) ≈ max((shared_norm === :linf ? norm(u, 1) : norm(u)) / r, norm(u, Inf)) / 1.7
      end
    end
    for c in (0.0, 0.3, 2.0), d in (0.0, 0.2, 4.0)
      u = [3.0, -1, 0.2]
      v = HDM._dirty_l2_prox!(copy(u), c, d)
      z = u - v
      @test norm(z) <= c + 1e-14 && norm(z, Inf) <= d + 1e-14
      (iszero(c) || iszero(d)) && @test v == u
    end
    # a nonfinite row fails every interval of the l2 threshold search
    @test_throws ErrorException HDM._dirty_threshold([NaN, 1.0], 1.2, :l2)
  end

  @testset "Clarabel references and certified paths" begin
    root = joinpath(@__DIR__, "data", "multitask_dirty")
    for name in sort(filter(d -> isdir(joinpath(root, d)), readdir(root)))
      @testset "$name" begin
        dir = joinpath(root, name)
        settings = read_values(joinpath(dir, "settings.csv"))
        q, L = parse(Int, settings["tasks"]), parse(Int, settings["levels"])
        shared = parse(Bool, settings["shared"])
        Xs = shared ? fill(read_matrix(dir, "X.csv"), q) : [read_matrix(dir, "X$t.csv") for t in 1:q]
        ys = [vec(read_matrix(dir, "y$t.csv")) for t in 1:q]
        ω = vec(read_matrix(dir, "penalty_factors.csv"))
        intercept, standardize_x = parse(Bool, settings["intercept"]), parse(Bool, settings["standardize_x"])
        shared_norm, task_weights = Symbol(settings["shared_norm"]), Symbol(settings["task_weights"])
        scales0 = scales(Xs, intercept, standardize_x)
        kw = (; intercept, standardize_x, shared_norm, task_weights, penalty_factors = ω)
        levels = [read_values(joinpath(dir, "level$k.csv")) for k in 1:L]
        for k in 1:L
          lb, ls = parse(Float64, levels[k]["lambda_shared"]), parse(Float64, levels[k]["lambda_specific"])
          sol = dirty_multitask(Xs, ys, lb, ls, TIGHT; kw...)
          check_solution(sol, Xs, ys, scales0)
          B, β0 = read_matrix(dir, "B$k.csv"), vec(read_matrix(dir, "intercept$k.csv"))
          ref = parse(Float64, levels[k]["objective"])
          @test abs(sol.objective - ref) <= 1e-9 * abs(ref)
          # fitted values are unique, but an objective error δ only bounds them to order
          # sqrt(2nδ): about 1e-6 at the fixtures' certified tolerance of 1e-12
          prediction_error = maximum(maximum(abs, predict(sol, Xs)[t] - (β0[t] .+ Xs[t] * B[:, t])) for t in 1:q)
          if prediction_error > 1e-5
            @info "Fixture comparison" case = name level = k prediction_error objective_difference = sol.objective - ref kkt = sol.kkt
          end
          @test prediction_error <= 1e-5
          for t in 1:q
            on = scales0[:, t] .> 0
            Xc = Xs[t][:, on] .- (intercept ? mean(Xs[t][:, on]; dims = 1) : 0)
            if rank(Xc) == count(on)
              @test maximum(abs, Matrix(sol.B)[on, t] - B[on, t]) <= 1e-5
            end
          end
          if shared
            ss = dirty_multitask(Xs[1], hcat(ys...), lb, ls, TIGHT; kw...)
            @test ss.B == sol.B && ss.intercept == sol.intercept
            @test predict(ss, Xs[1]) ≈ hcat(predict(sol, Xs)...)
          end
        end
        r = parse(Float64, levels[1]["lambda_shared"]) / parse(Float64, levels[1]["lambda_specific"])
        grid = all(>(0), ω) ? DirtyMultitaskPath(Xs, ys, TIGHT; ratio = r, nλ = 6, λminratio = 0.04, kw...) :
          DirtyMultitaskPath(Xs, ys, [0.3, 0.15, 0.08, 0.04], TIGHT; ratio = r, kw...)
        unscreened = DirtyMultitaskPath(Xs, ys, grid.λpath, TIGHT; ratio = r, screening = false, kw...)
        @test all(>(0), grid.kkt_checks) && all(iszero, unscreened.kkt_checks)
        for path in (grid, unscreened), k in eachindex(path.λpath)
          sol = HDM._dirty_solution(path, k)
          check_solution(sol, Xs, ys, scales0)
        end
        for k in eachindex(grid.λpath)
          warm = predict(HDM._dirty_solution(grid, k), Xs)
          off = predict(HDM._dirty_solution(unscreened, k), Xs)
          cold = predict(dirty_multitask(Xs, ys, r * grid.λpath[k], grid.λpath[k], TIGHT; kw...), Xs)
          @test all(maximum(abs, warm[t] - off[t]) <= 1e-7 for t in 1:q)
          @test all(maximum(abs, warm[t] - cold[t]) <= 1e-7 for t in 1:q)
        end
        if all(>(0), ω)
          c = task_weights === :equal ? ones(q) : q .* length.(ys) ./ sum(length.(ys))
          top = lambda_max(Xs, ys, scales0, c, ω, r, shared_norm; intercept)
          @test first(grid.λpath) / (1 + sqrt(eps())) ≈ top rtol = 1e-12
          edge = DirtyMultitaskPath(Xs, ys, [top * (1 + 1e-8), 0.99top], TIGHT; ratio = r, kw...)
          @test all(iszero, edge.Bpath[1])
          @test norm(edge.Bpath[2]) > 0
        end
      end
    end
  end

  rng = Xoshiro(90)
  n, p, q = 36, 7, 3
  X = centered_rand(rng, n, p) .* (0.5 .+ rand(rng, p))' .+ 0.5
  truth = zeros(p, q); truth[1:2, :] .= [1.0, -0.8]; truth[3, 1] = 1.2; truth[4, 2] = -1.0
  Y = X * truth .+ [1.0 -0.5 2.0] .+ 0.1 .* centered_rand(rng, n, q)
  Xs, ys = fill(X, q), [Y[:, t] for t in 1:q]

  @testset "Reductions and StatsAPI" begin
    c, ω, λ = [0.7, 1.3, 2.0], [0.4, 1.7, 0.0, 1, 1, 1, 1], 0.08
    for shared_norm in (:linf, :l2), r in (0.6, 1.0)
      sol = dirty_multitask(Xs, ys, r * λ, λ, TIGHT; shared_norm, task_weights = c, penalty_factors = ω)
      row = multitask_lasso(Xs, ys, r * λ, EXACT; row_norm = shared_norm, task_weights = c, penalty_factors = ω)
      @test maximum(abs, Matrix(sol.B - row.B)) <= 1e-7
      @test sol.intercept ≈ row.intercept atol = 1e-7
      @test all(iszero, sol.S)
    end
    for shared_norm in (:linf, :l2)
      for r in (shared_norm === :linf ? q : sqrt(q), q + 1)
        sol = dirty_multitask(Xs, ys, r * λ, λ, TIGHT; shared_norm, task_weights = c)
        entries = multitask_lasso(Xs, ys, λ, TIGHT; alpha = 1, task_weights = c)
        @test maximum(abs, Matrix(sol.B - entries.B)) <= 1e-7
        @test sol.intercept ≈ entries.intercept atol = 1e-7
        @test all(iszero, sol.A)
      end
      for (lb, ls) in ((0.2, 0.08), (0.03, 0.08))
        one = dirty_multitask(Xs[1:1], ys[1:1], lb, ls, TIGHT; shared_norm, task_weights = c[1:1])
        scalar = lasso(X, ys[1], min(lb, ls) / c[1], EXACT; intercept = true, standardize_x = true, gram = false)
        @test Vector(one.B[:, 1]) ≈ Vector(scalar.x) atol = 1e-7
        @test one.intercept[1] ≈ scalar.intercept atol = 1e-7
      end
      direct = dirty_multitask(X, Y, 0.12, 0.08, TIGHT; shared_norm)
      continuation = dirty_multitask(X, Y, 0.12, 0.08, CDOptions(TIGHT; warm_start = false, num_steps = 3); shared_norm)
      @test Matrix(direct.B) ≈ Matrix(continuation.B) atol = 1e-7
      @test size(coef(direct)) == (p + 1, q) && nobs(direct) == n * q
      @test dof(direct) == nnz(direct.B) + q
      @test fit(DirtyMultitaskSolution, X, Y, 0.12, 0.08, TIGHT; shared_norm).B == direct.B
      path = fit(DirtyMultitaskPath, X, Y, [0.2, 0.1], TIGHT; ratio = 1.5, shared_norm)
      @test coef(path, 0.1) == last(coef(path))
      @test predict(path, X)[end] ≈ hcat([v[:, end] for v in predict(path, Xs)]...)
      @test nobs(path) == n * q && dof(path)[end] == nnz(last(path.Bpath)) + q
      @test_throws ArgumentError coef(path, 0.15)
    end
  end

  @testset "Boundaries, sparse designs and certificates" begin
    for shared_norm in (:linf, :l2)
      for intercept in (false, true), (lb, ls) in ((0.0, 0.1), (0.1, 0.0), (0.0, 0.0))
        sol = dirty_multitask(X, Y, lb, ls, TIGHT; shared_norm, intercept)
        lsfit = (intercept ? [ones(n) X] : X) \ Y
        @test Matrix(coef(sol)) ≈ lsfit atol = 1e-7
        @test isnan(sol.gap)
        @test iszero(lb) ? all(iszero, sol.S) : all(iszero, sol.A)
        check_solution(sol, Xs, ys, scales(Xs, intercept, true))
      end
      Xz = [copy(X) for _ in 1:q]
      Xz[1][:, 7] .= 0; Xz[2][:, 7] .= 2
      for intercept in (false, true)
        kw = (; intercept, shared_norm, task_weights = [1.0, 2.0, 0.7])
        dense = dirty_multitask(Xz, ys, 0.12, 0.08, TIGHT; kw...)
        sp = dirty_multitask(sparse.(Xz), ys, 0.12, 0.08, TIGHT; kw...)
        @test Matrix(dense.B) ≈ Matrix(sp.B) atol = 1e-7
        check_solution(sp, sparse.(Xz), ys, scales(Xz, intercept, true))
      end
      unpen = [0.0; ones(p - 1)]
      sol = dirty_multitask(X, Y, 0.12, 0.08, TIGHT; shared_norm, penalty_factors = unpen)
      @test sol.converged && isnan(sol.gap)
      @test_throws ArgumentError DirtyMultitaskPath(X, Y; ratio = 1.5, shared_norm, penalty_factors = unpen)
      exhausted = DirtyMultitaskPath(X, Y, [0.2, 0.1], CDOptions(; max_iter = 0); ratio = 1.5, shared_norm)
      @test !any(exhausted.converged)
      failed = dirty_multitask(X, Y, 0.015, 0.01, CDOptions(; max_iter = 1, opt_tol = 0, gaptol = 1e-12); shared_norm)
      @test !failed.converged && failed.kkt > 1e-6
      floor = dirty_multitask(X, Y, 0.12, 0.08, CDOptions(; max_iter = 4, opt_tol = 1, gaptol = 1e-30); shared_norm)
      @test !floor.converged
      zero = dirty_multitask(X, fill(2.0, n, q), 0.12, 0.08; shared_norm)
      @test zero.converged && all(iszero, zero.B) && isnan(zero.gap)
      f32 = dirty_multitask(Float32.(X), Float32.(Y), 0.12f0, 0.08f0; shared_norm)
      @test eltype(f32.B) === Float32 && f32.converged
      @test Matrix(f32.B) ≈ Matrix(dirty_multitask(X, Y, 0.12, 0.08, TIGHT; shared_norm).B) atol = 2e-5
      scaled = dirty_multitask(1e8 .* X, Y, 0.12, 0.08; shared_norm, standardize_x = false)
      ref = dirty_multitask(1e8 .* X, Y, 0.12, 0.08, TIGHT; shared_norm, standardize_x = false)
      @test predict(scaled, 1e8 .* X) ≈ predict(ref, 1e8 .* X) atol = 1e-5
    end
  end

  @testset "CV leakage, common folds and threads" begin
    Xt = [X[1:30, :], X .+ 0.4, 1.3 .* X[1:33, :]]
    yt = [ys[1][1:30], ys[2], ys[3][1:33]]
    counts = [8 10 12; 12 14 10; 13 9 11]
    ft = [vcat([fill(k, counts[t, k]) for k in 1:3]...) for t in 1:q]
    ratios = [1.8, 1.0]
    for shared_norm in (:linf, :l2)
      kw = (; ratios, shared_norm, nλ = 4, λminratio = 0.1, task_weights = :observations)
      cv = CVDirtyMultitaskPath(Xt, yt, TIGHT; folds = ft, kw..., rng = Xoshiro(8), threads = false)
      parallel = CVDirtyMultitaskPath(Xt, yt, TIGHT; folds = ft, kw..., rng = Xoshiro(8), threads = true)
      @test cv.cvm == parallel.cvm && cv.cvse == parallel.cvse && cv.selected == parallel.selected
      @test cv.folds == ft && size(cv.cvm) == (2, 4)
      for i in eachindex(ratios)
        sse = zeros(q, 3, length(cv.paths[i].λpath))
        for k in 1:3
          train, test = [findall(!=(k), f) for f in ft], [findall(==(k), f) for f in ft]
          fp = DirtyMultitaskPath([Xt[t][train[t], :] for t in 1:q], [yt[t][train[t]] for t in 1:q],
                                  cv.paths[i].λpath, TIGHT; ratio = ratios[i], shared_norm, task_weights = :observations)
          pred = predict(fp, [Xt[t][test[t], :] for t in 1:q])
          for t in 1:q, j in eachindex(fp.λpath)
            sse[t, k, j] = sum(abs2, yt[t][test[t]] - pred[t][:, j])
          end
        end
        per_task = dropdims(sum(sse; dims = 2); dims = 2) ./ length.(yt)
        @test cv.cvm_task[:, i, :] ≈ per_task atol = 1e-10
        @test cv.cvm[i, :] ≈ vec(sum(per_task .* cv.task_weights; dims = 1)) / sum(cv.task_weights) atol = 1e-10
      end
      candidates = [(cv.cvm[i, j], -ratios[i], -cv.paths[i].λpath[j], CartesianIndex(i, j)) for i in 1:2 for j in 1:4]
      @test cv.selected == last(minimum(candidates))
      @test cv.solution.converged && coef(cv) == coef(cv.solution)
      @test predict(cv, Xt) == predict(cv.solution, Xt) && nobs(cv) == sum(length.(yt))
      @test dof(cv) == dof(cv.solution)
      folds = repeat(1:3; inner = 12)
      shared = CVDirtyMultitaskPath(X, Y, TIGHT; folds, ratios, shared_norm, nλ = 4, threads = false)
      tasks = CVDirtyMultitaskPath(Xs, ys, TIGHT; folds = fill(folds, q), ratios, shared_norm, nλ = 4, threads = false)
      @test shared.folds == fill(folds, q)
      @test shared.cvm ≈ tasks.cvm atol = 1e-10
      @test predict(shared, X) == predict(shared.solution, X)
      @test fit(CVDirtyMultitaskPath, X, Y, TIGHT; folds, ratios, shared_norm, nλ = 2).solution.converged
      @test_throws ArgumentError CVDirtyMultitaskPath(X, Y, CDOptions(; max_iter = 0); folds, ratios, shared_norm, nλ = 3)
      # Exact ties select the larger ratio, then the larger level.
      tie = CVDirtyMultitaskPath(X, fill(2.0, n, q), TIGHT; folds, ratios, shared_norm, nλ = 3)
      @test tie.ratio == maximum(ratios) && tie.λmin == first(tie.path.λpath)
    end
    # A signal present only in training folds causes early fold-path stops.
    block = [1.0 1; 1 -1; -1 1; -1 -1]
    xp = repeat(block, 3, 1)
    yp = repeat(2 .* block[:, 1], 3)
    labels = repeat(1:3; inner = 4)
    short = CDOptions(; max_iter = 1, opt_tol = 1e-13, gaptol = 0)
    nc = CVDirtyMultitaskPath(xp, reshape(yp, :, 1), short; folds = labels, ratios = [1.0, 2.0], nλ = 4)
    @test all(isnan, nc.cvm[:, 2:end]) && nc.λmin == first(nc.path.λpath)
    yp = vcat([2 .* block[:, 1] .+ a .* block[:, 2] for a in (1, 2, -3)]...)
    partial = CVDirtyMultitaskPath(xp, reshape(yp, :, 1), TIGHT; folds = labels, ratios = [0.7, 1.2], nλ = 6, max_rows = 1)
    @test any(isnan, partial.cvm)
    @test isfinite(partial.cvm[partial.selected])
    padded = CVDirtyMultitaskPath(xp, reshape(repeat(2 .* block[:, 1], 3), :, 1), TIGHT;
                                 folds = labels, ratios = [0.7, 1.2], nλ = 6, max_rows = 0)
    @test size(padded.cvm) == (2, 6) && all(isnan, padded.cvm[:, 3:end])
    @test all(length(path.λpath) == 2 for path in padded.paths)
    default = CVDirtyMultitaskPath(X, Y, TIGHT; folds = repeat(1:3; inner = 12), nλ = 2, shared_norm = :l2)
    @test default.ratios ≈ exp.(range(0, log(sqrt(q)); length = 5))
  end

  @testset "Input checks" begin
    for (lb, ls) in ((-0.1, 0.1), (0.1, -0.1), (Inf, 1.0), (1.0, NaN))
      @test_throws ArgumentError dirty_multitask(X, Y, lb, ls)
    end
    for ratio in (0, -1, Inf, NaN)
      @test_throws ArgumentError DirtyMultitaskPath(X, Y; ratio)
      @test_throws ArgumentError CVDirtyMultitaskPath(X, Y; ratios = [ratio])
    end
    for levels in ([0.1, 0.2], [-1.0], [Inf], [NaN])
      @test_throws ArgumentError DirtyMultitaskPath(X, Y, levels; ratio = 1.5)
    end
    @test_throws ArgumentError dirty_multitask(X, Y, 0.1, 0.1; shared_norm = :other)
    @test_throws ArgumentError DirtyMultitaskPath(X, Y; ratio = 1.5, nλ = 0)
    @test_throws ArgumentError DirtyMultitaskPath(X, Y; ratio = 1.5, λminratio = 2)
    @test_throws DimensionMismatch dirty_multitask(X, Y[2:end, :], 0.1, 0.1)
    @test_throws DimensionMismatch dirty_multitask(Xs, ys[1:2], 0.1, 0.1)
    @test_throws ArgumentError dirty_multitask(X, Y, 0.1, 0.1; task_weights = [1, -1, 1])
    @test_throws ArgumentError dirty_multitask(X, Y, 0.1, 0.1; penalty_factors = zeros(p))
    @test_throws DimensionMismatch dirty_multitask(X, Y, 0.1, 0.1; penalty_factors = ones(p - 1))
    @test_throws ArgumentError dirty_multitask(X, Y, 0.1, 0.1; warm_start = true)
    @test_throws DimensionMismatch predict(dirty_multitask(X, Y, 0.1, 0.1), X[:, 1:2])
    @test_throws ArgumentError CVDirtyMultitaskPath(X, Y; ratios = [])
    @test_throws ArgumentError CVDirtyMultitaskPath(X, Y; nfolds = 1)
    @test_throws ArgumentError CVDirtyMultitaskPath(X, Y; folds = zeros(Int, n))
    @test_throws DimensionMismatch CVDirtyMultitaskPath(X, Y; folds = ones(Int, n - 1))
    @test_throws ArgumentError CVDirtyMultitaskPath(X[1:2, :], Y[1:2, :]; folds = [1, 2])
  end
end

end
