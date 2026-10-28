module MultitaskPenaltiesTest

using HighDimensionalModels
using DelimitedFiles, LinearAlgebra, Random, SparseArrays, Statistics, Test

const HDM = HighDimensionalModels
const TIGHT = CDOptions(; opt_tol = 1e-13, gaptol = 1e-12)
const EXACT = CDOptions(; opt_tol = 1e-14, gaptol = 0)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

# The column scales and loss gradient, computed on the original data.
function scales(Xs, intercept, standardize)
  S = zeros(size(Xs[1], 2), length(Xs))
  for t in eachindex(Xs), j in axes(S, 1)
    x = Vector(Xs[t][:, j])
    (intercept ? all(==(x[1]), x) : all(iszero, x)) && continue
    S[j, t] = standardize ? sqrt(sum(abs2, x .- (intercept ? mean(x) : 0)) / length(x)) : 1
  end
  S
end

function gradient(Xs, ys, B, β0, c)
  hcat([-c[t] .* (Xs[t]' * (ys[t] .- β0[t] .- Xs[t] * B[:, t])) ./ length(ys[t]) for t in eachindex(ys)]...)
end

function objective(Xs, ys, B, β0, λ, S, c, ω, row_norm, alpha)
  U = S .* B
  loss = sum(c[t] * sum(abs2, ys[t] .- β0[t] .- Xs[t] * B[:, t]) / (2 * length(ys[t])) for t in eachindex(ys))
  loss + λ * sum(row_norm === :linf ? ω[j] * norm(U[j, :], Inf) :
                 alpha * sum(abs, U[j, :]) + (1 - alpha) * ω[j] * norm(U[j, :]) for j in axes(B, 1))
end

# Optimality for zero and nonzero rows, including the intercept conditions.
function kkt(Xs, ys, B, β0, λ, S, c, ω, row_norm, alpha; intercept = true)
  G = gradient(Xs, ys, B, β0, c)
  err = 0.0
  for t in eachindex(ys)
    r = ys[t] .- β0[t] .- Xs[t] * B[:, t]
    intercept && (err = max(err, c[t] * abs(mean(r))))
  end
  for j in axes(B, 1)
    on = S[j, :] .> 0
    any(!iszero, B[j, .!on]) && return Inf
    err = max(err, maximum(abs, G[j, .!on]; init = 0.0))
    u, g = S[j, on] .* B[j, on], G[j, on] ./ S[j, on]
    isempty(u) && continue
    if row_norm === :linf
      if iszero(norm(u))
        err = max(err, sum(abs, g) - λ * ω[j])
      else
        top = norm(u, Inf)
        ties = abs.(abs.(u) .- top) .<= 1e-9 * max(1, top)
        err = max(err, abs(sum(abs, g) - λ * ω[j]), norm(g[.!ties]),
                  maximum(g .* sign.(u); init = 0.0))
      end
    elseif iszero(norm(u))
      err = max(err, norm(max.(abs.(g) .- alpha * λ, 0)) - (1 - alpha) * λ * ω[j])
    else
      z = g .+ (1 - alpha) * λ * ω[j] .* u ./ norm(u)
      for t in eachindex(z)
        err = max(err, iszero(u[t]) ? abs(z[t]) - alpha * λ : abs(z[t] + alpha * λ * sign(u[t])))
      end
    end
  end
  err
end

function check_path(path, Xs, ys, S, c, ω)
  G0 = gradient(Xs, ys, zeros(size(S)), path.intercept ? mean.(ys) : zeros(length(ys)), c)
  for i in eachindex(G0)
    G0[i] = iszero(S[i]) ? 0 : G0[i] / S[i]
  end
  scale = max(1, maximum(norm(G0[j, :]) for j in axes(S, 1)))
  for k in eachindex(path.λpath)
    @test path.converged[k]
    @test kkt(Xs, ys, Matrix(path.Bpath[k]), path.β0path[k], path.λpath[k], S, c, ω,
              path.row_norm, path.alpha; intercept = path.intercept) <= 1e-6 * scale
  end
end

read_values(file) = Dict(row[1] => row[2] for row in eachrow(readdlm(file, ',', String)))
read_matrix(dir, file) = readdlm(joinpath(dir, file), ',', Float64)

# A fit and its fitted values, with the same coordinate orders at every scale.
function scaled_fit(Xs, ys, λ, options::CDOptions = CDOptions(); shared, intercept, row_norm, alpha)
  if shared
    sol = multi_lasso(Xs[1], hcat(ys...), λ, options;
                      intercept, standardize_x = false, row_norm, alpha, rng = Xoshiro(3))
    return sol, collect(eachcol(predict(sol, Xs[1])))
  end
  sol = multitask_lasso(Xs, ys, λ, options; intercept, standardize_x = false, row_norm, alpha,
                        rng = Xoshiro(3))
  sol, predict(sol, Xs)
end

# The weighted RMS norm across tasks, in the units of the responses.
response_norm(ys, c) = sqrt(sum(c[t] * sum(abs2, ys[t]) / length(ys[t]) for t in eachindex(ys)))

@testset "Multitask row penalties" begin
  @testset "Clarabel reference and screening" begin
    root = joinpath(@__DIR__, "data", "multitask")
    cases = sort(filter(d -> isdir(joinpath(root, d)), readdir(root)))
    for name in cases
      dir = joinpath(root, name)
      s = read_values(joinpath(dir, "settings.csv"))
      q, L = parse(Int, s["tasks"]), parse(Int, s["levels"])
      shared = parse(Bool, s["shared"])
      Xs = shared ? fill(read_matrix(dir, "X.csv"), q) : [read_matrix(dir, "X$t.csv") for t in 1:q]
      ys = [vec(read_matrix(dir, "y$t.csv")) for t in 1:q]
      ω = vec(read_matrix(dir, "penalty_factors.csv"))
      intercept, standardize_x = parse(Bool, s["intercept"]), parse(Bool, s["standardize_x"])
      row_norm, alpha = Symbol(s["row_norm"]), parse(Float64, s["alpha"])
      task_weights = Symbol(s["task_weights"])
      n = length.(ys)
      c = task_weights === :equal ? ones(q) : q .* n ./ sum(n)
      S = scales(Xs, intercept, standardize_x)
      kw = (; intercept, standardize_x, row_norm, alpha, task_weights, penalty_factors = ω)
      levels = [read_values(joinpath(dir, "level$k.csv")) for k in 1:L]
      λs = [parse(Float64, l["lambda"]) for l in levels]
      for k in 1:L
        sol = multitask_lasso(Xs, ys, λs[k], TIGHT; kw..., rng = Xoshiro(1))
        B, β0 = read_matrix(dir, "B$k.csv"), vec(read_matrix(dir, "intercept$k.csv"))
        obj = objective(Xs, ys, Matrix(sol.B), sol.intercept, λs[k], S, c, ω, row_norm, alpha)
        ref = parse(Float64, levels[k]["objective"])
        @test sol.converged
        @test abs(obj - ref) <= 1e-9 * abs(ref)
        @test maximum(abs, Matrix(sol.B) - B) <= 1e-6
        @test all(maximum(abs, predict(sol, Xs)[t] - (β0[t] .+ Xs[t] * B[:, t])) <= 1e-6 for t in 1:q)
        if shared
          ms = multi_lasso(Xs[1], hcat(ys...), λs[k], TIGHT; intercept, standardize_x, row_norm, alpha)
          @test maximum(abs, Matrix(ms.B) - B) <= 1e-6
          @test nobs(ms) == n[1] && ms.row_norm === row_norm && ms.alpha == alpha
        end
      end
      grid = MultitaskLassoPath(Xs, ys, TIGHT; nλ = 9, λminratio = 0.02, kw..., rng = Xoshiro(2))
      λmax = parse(Float64, s["lambda_max"])
      @test first(grid.λpath) / (1 + sqrt(eps())) ≈ λmax rtol = 1e-12
      edge = MultitaskLassoPath(Xs, ys, [λmax * (1 + 1e-8), 0.99λmax], TIGHT; kw...)
      @test nnz(edge.Bpath[1]) == 0 && nnz(edge.Bpath[2]) > 0
      unscreened = MultitaskLassoPath(Xs, ys, grid.λpath, TIGHT; kw..., screening = false, rng = Xoshiro(3))
      @test all(maximum(abs, Matrix(a - b)) <= 1e-7 for (a, b) in zip(grid.Bpath, unscreened.Bpath))
      @test all(>(0), grid.kkt_checks) && all(iszero, unscreened.kkt_checks)
      check_path(grid, Xs, ys, S, c, ω)
      check_path(unscreened, Xs, ys, S, c, ω)
      d = HDM._MultitaskData(Xs, ys; intercept, standardize_x, task_weights, stacked = false)
      @test d.Z === nothing && d.ỹ === nothing
      shared && @test all(X -> X === d.X[1], d.X)
    end
  end

  @testset "Change rule in fitted-value units" begin
    rng = Xoshiro(1)
    Z = rand(rng, 50, 6) .- 0.5
    Z[:, 2] = Z[:, 1] .+ 0.05 .* (rand(rng, 50) .- 0.5)
    truth = [1.0 2 -1; 1 0.5 1; 0 0 0; 0.5 0 0; 0 0 0; 0 0 0]
    Y = Z * truth .+ 0.1 .* (rand(rng, 50, 3) .- 0.5)
    Zs = [Z, Z .+ 0.1 .* (rand(rng, 50, 6) .- 0.5), Z[1:45, :]]
    task_y = [Zs[t] * truth[:, t] .+ 0.1 .* (rand(rng, size(Zs[t], 1)) .- 0.5) for t in 1:3]
    tight = CDOptions(; opt_tol = 1e-14, gaptol = 1e-12)
    for shared in (true, false), intercept in (false, true), scale in (1.0, 1e8)
      Xs = shared ? fill(scale .* Z, 3) : [scale .* X for X in Zs]
      ys = shared ? collect(eachcol(Y)) : task_y
      c = ones(3)
      r = response_norm([y .- (intercept ? mean(y) : 0) for y in ys], c)
      for (row_norm, alpha) in ((:linf, 0.0), (:l2, 0.5))
        @testset "shared=$shared intercept=$intercept scale=$scale row_norm=$row_norm" begin
          kw = (; shared, intercept, row_norm, alpha)
          sol, fitted = scaled_fit(Xs, ys, 0.01; kw...)
          ref, reference = scaled_fit(Xs, ys, 0.01, tight; kw...)
          @test sol.converged && ref.converged
          @test sol.iterations > 1
          # Prediction error is at most 1e-5 of the centered response RMS scale.
          @test response_norm(fitted .- reference, c) <= 1e-5 * r
          # Scaling Y and λ together scales the minimizer of this objective.
          large, large_fit = scaled_fit(Xs, [1e6 .* y for y in ys], 1e6 * 0.01; kw...)
          @test large.converged
          @test response_norm([large_fit[t] ./ 1e6 .- fitted[t] for t in 1:3], c) <= 1e-5 * r
        end
      end
    end
    for (row_norm, alpha) in ((:linf, 0.0), (:l2, 0.5))
      # A zero centered response has an exact zero-coefficient solution.
      for intercept in (false, true)
        zero = multi_lasso(Z, fill(intercept ? 2.0 : 0.0, 50, 3), 0.01;
                            intercept, standardize_x = false, row_norm, alpha)
        @test zero.converged && zero.iterations == 1 && all(iszero, zero.B)
        exhausted = multi_lasso(Z, zeros(50, 3), 0.01, CDOptions(; max_iter = 0);
                                 intercept, standardize_x = false, row_norm, alpha)
        @test !exhausted.converged && exhausted.iterations == 0
      end
      # Absolute tolerances still measure changes of U itself.
      absolute = multi_lasso(1e8 .* Z, Y, 0.01, CDOptions(; relative_tol = false);
                              standardize_x = false, row_norm, alpha, rng = Xoshiro(3))
      @test absolute.converged && absolute.iterations == 1
    end
  end

  rng = Xoshiro(89)
  n, p, q = 36, 8, 3
  X = centered_rand(rng, n, p) .* (0.5 .+ rand(rng, p))' .+ 0.4
  B = zeros(p, q); B[1:2, :] .= 1; B[3, 1] = -1.5; B[4, 2] = 1.2
  Y = X * B .+ [1.0 2.0 -1.0] .+ 0.1 .* centered_rand(rng, n, q)
  Xs, ys = fill(X, q), [Y[:, t] for t in 1:q]

  @testset "Reductions and compatibility" begin
    λ, c = 0.08, [0.7, 1.4, 2.0]
    for (row_norm, alpha) in ((:linf, 0.0), (:l2, 0.4))
      one = multitask_lasso(Xs[1:1], ys[1:1], λ, TIGHT; row_norm, alpha, task_weights = c[1:1])
      scalar = lasso(X, ys[1], λ / c[1], EXACT; intercept = true, standardize_x = true, gram = false)
      @test maximum(abs, vec(Matrix(one.B)) - Vector(scalar.x)) <= 1e-7
      @test one.intercept[1] ≈ scalar.intercept atol = 1e-7
      mt = multitask_lasso(Xs, ys, λ, TIGHT; row_norm, alpha, task_weights = fill(1.7, q))
      multi = multi_lasso(X, Y, λ / 1.7, TIGHT; row_norm, alpha)
      @test maximum(abs, Matrix(mt.B - multi.B)) <= 1e-7
      @test mt.intercept ≈ multi.intercept atol = 1e-7
      cold = multitask_lasso(Xs, ys, λ, CDOptions(TIGHT; warm_start = false, num_steps = 4); row_norm, alpha)
      warm = multitask_lasso(Xs, ys, λ, TIGHT; row_norm, alpha)
      @test maximum(abs, Matrix(cold.B - warm.B)) <= 1e-7
      @test cold.iterations > warm.iterations
      cyclic = multitask_lasso(Xs, ys, λ, CDOptions(TIGHT; order = :cyclic); row_norm, alpha)
      @test maximum(abs, Matrix(cyclic.B - warm.B)) <= 1e-7
    end
    entries = multitask_lasso(Xs, ys, λ, TIGHT; alpha = 1, task_weights = c, penalty_factors = collect(1.0:p))
    for t in 1:q
      scalar = lasso(X, ys[t], λ / c[t], EXACT; intercept = true, standardize_x = true, gram = false)
      @test maximum(abs, Vector(entries.B[:, t]) - Vector(scalar.x)) <= 1e-7
    end
    a = multitask_lasso(Xs, ys, λ, EXACT; rng = Xoshiro(9))
    b = multitask_lasso(Xs, ys, λ, EXACT; row_norm = :l2, alpha = 0, rng = Xoshiro(9))
    @test a.B == b.B && a.intercept == b.intercept && a.iterations == b.iterations
    default_task = a
    ma = multi_lasso(X, Y, λ, EXACT; rng = Xoshiro(9))
    mb = multi_lasso(X, Y, λ, EXACT; row_norm = :l2, alpha = 0, rng = Xoshiro(9))
    @test ma.B == mb.B && ma.intercept == mb.intercept && ma.iterations == mb.iterations
    for constructor in (MultitaskLassoPath, MultiLassoPath)
      data = constructor === MultitaskLassoPath ? (Xs, ys) : (X, Y)
      a = constructor(data..., EXACT; nλ = 5, rng = Xoshiro(2))
      b = constructor(data..., EXACT; nλ = 5, row_norm = :l2, alpha = 0, rng = Xoshiro(2))
      @test a.λpath == b.λpath && a.Bpath == b.Bpath && a.iterations == b.iterations
    end
    d = HDM._MultitaskData([X, X .* 1.3, X[:, :] .- 0.7], ys;
                            intercept = true, standardize_x = true, task_weights = c, stacked = false)
    internal = HDM._row_path(d, ones(p), [0.4, 0.15, λ], TIGHT)
    existing = MultitaskLassoPath(d.X, ys, internal.λpath, EXACT; task_weights = c)
    @test all(maximum(abs, Matrix(a - b)) <= 1e-7 for (a, b) in zip(internal.Bpath, existing.Bpath))
    check_path(internal, d.X, ys, scales(d.X, true, true), c, ones(p))
    # Both positional forms from before the metadata fields remain available.
    for sol in (default_task, ma)
      fields = [getfield(sol, k) for k in 1:fieldcount(typeof(sol))-2]
      old = Base.typename(typeof(sol)).wrapper(fields...)
      typed = typeof(sol)(fields...)
      @test old.row_norm === :l2 && iszero(old.alpha) && typed.B == sol.B
    end
    for path in (existing, MultiLassoPath(X, Y; nλ = 3))
      fields = [getfield(path, k) for k in 1:fieldcount(typeof(path))-2]
      old = Base.typename(typeof(path)).wrapper(fields...)
      typed = typeof(path)(fields...)
      @test old.row_norm === :l2 && iszero(old.alpha) && typed.Bpath == path.Bpath
    end
  end

  @testset "Boundary cases" begin
    for (row_norm, alpha) in ((:linf, 0.0), (:l2, 0.5))
      for intercept in (false, true)
        sol = multi_lasso(X, Y, 0.0, EXACT; intercept, standardize_x = false, row_norm, alpha)
        ls = (intercept ? [ones(n) X] : X) \ Y
        @test sol.converged && maximum(abs, Matrix(coef(sol)) - ls) <= 1e-7
      end
      Xz = copy(X); Xz[:, 7] .= 0; Xz[:, 8] .= 2
      sol = multi_lasso(Xz, Y, 0.1, TIGHT; row_norm, alpha)
      @test all(iszero, sol.B[7:8, :])
      @test multi_lasso(X, fill(2.0, n, q), 0.1, TIGHT; row_norm, alpha).B == spzeros(p, q)
      sparsefit = multitask_lasso(fill(sparse(Xz), q), ys, 0.1, TIGHT;
                                  intercept = false, row_norm, alpha, task_weights = [1.0, 2.0, 0.7])
      densefit = multitask_lasso(fill(Xz, q), ys, 0.1, TIGHT;
                                 intercept = false, row_norm, alpha, task_weights = [1.0, 2.0, 0.7])
      @test maximum(abs, Matrix(sparsefit.B - densefit.B)) <= 1e-7
      @test kkt(fill(sparse(Xz), q), ys, Matrix(sparsefit.B), sparsefit.intercept, 0.1,
                scales(fill(Xz, q), false, true), [1.0, 2.0, 0.7], ones(p), row_norm, alpha;
                intercept = false) <= 1e-6
      sparsecenter = multi_lasso(sparse(Xz), Y, 0.1, TIGHT; row_norm, alpha)
      @test maximum(abs, Matrix(sparsecenter.B - sol.B)) <= 1e-7
      Xe = [copy(X) for _ in 1:q]
      Xe[1][:, 1] .= 0; Xe[2][:, 1] .= 2
      absent = multitask_lasso(Xe, ys, 0.1, TIGHT; row_norm, alpha)
      @test all(iszero, absent.B[1, 1:2]) && !iszero(absent.B[1, 3])
      @test kkt(Xe, ys, Matrix(absent.B), absent.intercept, 0.1, scales(Xe, true, true),
                ones(q), ones(p), row_norm, alpha) <= 1e-6
      f32 = multi_lasso(Float32.(X), Float32.(Y), 0.1f0; row_norm, alpha)
      @test eltype(f32.B) === Float32 && f32.converged
      @test Matrix(f32.B) ≈ Matrix(multi_lasso(X, Y, 0.1, TIGHT; row_norm, alpha).B) atol = 2e-5
      exhausted = MultiLassoPath(X, Y, [0.2, 0.1], CDOptions(; max_iter = 0); row_norm, alpha)
      @test !any(exhausted.converged) && all(iszero, exhausted.iterations)
      failed = multi_lasso(X, Y, 0.01, CDOptions(; max_iter = 1, opt_tol = 0, gaptol = 1e-12); row_norm, alpha)
      @test !failed.converged && failed.iterations == 1
      cut = MultiLassoPath(X, Y; nλ = 10, max_rows = 0, row_norm, alpha)
      @test length(cut.λpath) == 2 && length(last(cut.activerows)) > 0
      weighted = [0.0; ones(p - 1)]
      unpen = multitask_lasso(Xs, ys, 0.1, EXACT; row_norm, alpha, penalty_factors = weighted)
      @test unpen.converged
      check_path(MultitaskLassoPath(Xs, ys, [0.2, 0.1], EXACT; row_norm, alpha, penalty_factors = weighted),
                  Xs, ys, scales(Xs, true, true), ones(q), weighted)
      if row_norm === :linf
        @test_throws ArgumentError MultitaskLassoPath(Xs, ys; row_norm, penalty_factors = weighted)
      else
        @test all(isfinite, MultitaskLassoPath(Xs, ys; alpha, penalty_factors = weighted, nλ = 3).λpath)
      end
    end
    # A gap below its rounding floor cannot certify convergence.
    @test !multi_lasso(X, Y, 0.1, CDOptions(; max_iter = 4, opt_tol = 1, gaptol = 1e-30); row_norm = :linf).converged
    @test multi_lasso(X, Y, 0.1, CDOptions(; max_iter = 4, opt_tol = 1, gaptol = 0); row_norm = :linf).converged
    for (rn, a) in ((:other, 0), (:linf, 0.5), (:l2, -0.1), (:l2, 1.1), (:l2, NaN))
      @test_throws ArgumentError multitask_lasso(Xs, ys, 0.1; row_norm = rn, alpha = a)
      @test_throws ArgumentError MultiLassoPath(X, Y; row_norm = rn, alpha = a)
    end
    for levels in ([0.1, 0.2], [Inf], [NaN], [-0.1])
      @test_throws ArgumentError MultitaskLassoPath(Xs, ys, levels; row_norm = :linf)
    end
    @test_throws ArgumentError MultiLassoPath(X, Y; row_norm = :linf, penalty_factors = ones(p))
    @test_throws ArgumentError MultiLassoPath(X, Y; alpha = 0.5, task_weights = ones(q))
  end

  @testset "Cross-validation" begin
    fold_sizes = [10, 12, 14]
    folds = vcat([fill(k, fold_sizes[k]) for k in 1:3]...)
    for (row_norm, alpha) in ((:linf, 0.0), (:l2, 0.4))
      cv = CVMultiLassoPath(X, Y, TIGHT; folds, nλ = 5, row_norm, alpha, rng = Xoshiro(5), threads = false)
      parallel = CVMultiLassoPath(X, Y, TIGHT; folds, nλ = 5, row_norm, alpha, rng = Xoshiro(5), threads = true)
      @test cv.cvm == parallel.cvm && cv.cvse == parallel.cvse && cv.λmin == parallel.λmin && cv.λ1se == parallel.λ1se
      sse = zeros(q, 3, length(cv.λpath))
      seeds = rand(Xoshiro(5), UInt64, 4)
      for k in 1:3
        train, test = findall(!=(k), folds), findall(==(k), folds)
        fp = MultiLassoPath(X[train, :], Y[train, :], cv.λpath, TIGHT; row_norm, alpha, rng = Xoshiro(seeds[k]))
        predictions = predict(fp, X[test, :])
        for j in eachindex(predictions), t in 1:q
          sse[t, k, j] = sum(abs2, Y[test, t] - predictions[j][:, t])
        end
      end
      response = dropdims(sum(sse; dims = 2); dims = 2) / n
      @test cv.cvm_response ≈ response atol = 1e-10
      @test cv.cvm ≈ vec(mean(response; dims = 1)) atol = 1e-10
      errors = dropdims(sum(sse; dims = 1); dims = 1) ./ (q .* fold_sizes)
      manual_se = sqrt.(vec(sum(fold_sizes .* (errors .- cv.cvm') .^ 2; dims = 1)) ./ (2n))
      @test cv.cvse ≈ manual_se atol = 1e-10
      @test predict(cv, X) ≈ predict(cv.path, X)[findfirst(==(cv.λ1se), cv.λpath)]
      @test coef(cv; λ = :λmin) == coef(cv.path, cv.λmin)
      @test fit(CVMultiLassoPath, X, Y, TIGHT; folds, nλ = 3, row_norm, alpha).path.row_norm === row_norm
      truncated = CVMultiLassoPath(X, Y, TIGHT; folds, nλ = 8, max_rows = 0, row_norm, alpha)
      @test length(truncated.λpath) == 2 && length(last(truncated.path.activerows)) > 0
      @test truncated.λmin in truncated.λpath && truncated.λ1se in truncated.λpath
      # Unequal task sizes refit the :observations weights in every fold.
      Xt = [X[1:30, :], X .+ 0.2, X[1:33, :] .* 1.3]
      yt = [ys[1][1:30], ys[2], ys[3][1:33]]
      counts = [8 10 12; 12 14 10; 13 9 11]
      ft = [vcat([fill(k, counts[t, k]) for k in 1:3]...) for t in 1:q]
      mt = CVMultitaskLassoPath(Xt, yt, TIGHT; folds = ft, nλ = 5, row_norm, alpha,
                                task_weights = :observations, threads = false, rng = Xoshiro(8))
      st = zeros(3, 3, length(mt.λpath))
      for k in 1:3
        train = [findall(!=(k), f) for f in ft]
        test = [findall(==(k), f) for f in ft]
        fp = MultitaskLassoPath([Xt[t][train[t], :] for t in 1:q], [yt[t][train[t]] for t in 1:q], mt.λpath,
                                TIGHT; row_norm, alpha, task_weights = :observations)
        predictions = predict(fp, [Xt[t][test[t], :] for t in 1:q])
        for t in 1:q, j in eachindex(mt.λpath)
          st[t, k, j] = sum(abs2, yt[t][test[t]] - predictions[t][:, j])
        end
      end
      per_task = dropdims(sum(st; dims = 2); dims = 2) ./ length.(yt)
      @test mt.cvm_task ≈ per_task atol = 1e-10
      @test mt.cvm ≈ vec(sum(per_task .* mt.task_weights; dims = 1)) / sum(mt.task_weights) atol = 1e-10
      weights = vec(sum(counts .* (mt.task_weights ./ length.(yt)); dims = 1))
      errors = dropdims(sum(st .* (mt.task_weights ./ length.(yt)); dims = 1); dims = 1) ./ weights
      manual_se = sqrt.(vec(sum(weights .* (errors .- mt.cvm') .^ 2; dims = 1)) ./ (2sum(weights)))
      @test mt.cvse ≈ manual_se atol = 1e-10
      mp = CVMultitaskLassoPath(Xt, yt, TIGHT; folds = ft, nλ = 5, row_norm, alpha,
                                task_weights = :observations, threads = true, rng = Xoshiro(8))
      @test mt.cvm == mp.cvm && mt.cvse == mp.cvse
      # No fold level is selectable when its solver never converges.
      @test_throws ArgumentError CVMultiLassoPath(X, Y, CDOptions(; max_iter = 0); folds, nλ = 3, row_norm, alpha)
      @test_throws ArgumentError CVMultitaskLassoPath(Xs, ys, CDOptions(; max_iter = 0);
                                                     folds = fill(folds, q), nλ = 3, row_norm, alpha)
      # Each training sample has a second signal that cancels in the full data.
      block = [1.0 1; 1 -1; -1 1; -1 -1]
      xp = repeat(block, 3, 1)
      yp = vcat([2 .* block[:, 1] .+ a .* block[:, 2] for a in (1, 2, -3)]...)
      labels = repeat(1:3; inner = 4)
      pcv = CVMultiLassoPath(xp, reshape(yp, :, 1), TIGHT; folds = labels, nλ = 8,
                             max_rows = 1, row_norm, alpha, threads = false)
      tcv = CVMultitaskLassoPath([xp], [yp], TIGHT; folds = [labels], nλ = 8,
                                 max_rows = 1, row_norm, alpha, threads = false)
      @test length(pcv.λpath) == 8 && any(isnan, pcv.cvm) && any(isnan, tcv.cvm)
      @test pcv.cvm ≈ tcv.cvm nans = true atol = 1e-10
      @test pcv.λmin == tcv.λmin && pcv.λ1se == tcv.λ1se
      partial = findall(j -> !isnan(pcv.cvm[j]), eachindex(pcv.λpath))
      @test length(partial) < 8 && pcv.λmin >= pcv.λpath[2]
      # Only the zero level converges with one pass on this homogeneous sample.
      homogeneous = repeat(2 .* block[:, 1], 3)
      short = CDOptions(; max_iter = 1, opt_tol = 1e-13, gaptol = 0)
      nc = CVMultiLassoPath(xp, reshape(homogeneous, :, 1), short; folds = labels, nλ = 4, row_norm, alpha)
      nt = CVMultitaskLassoPath([xp], [homogeneous], short; folds = [labels], nλ = 4, row_norm, alpha)
      @test all(isnan, nc.cvm[2:end]) && all(isnan, nt.cvm[2:end])
      @test nc.λmin == nc.λ1se == first(nc.λpath)
    end
    for standardize_x in (false, true)
      scalar = CVLassoPath(X, ys[1], EXACT; folds, nλ = 6, intercept = true, standardize_x, rng = Xoshiro(7), gram = false)
      for (row_norm, alpha) in ((:l2, 0.0), (:linf, 0.0), (:l2, 0.5))
        options = row_norm === :l2 && iszero(alpha) ? CDOptions(TIGHT; gaptol = 0) : TIGHT
        cv = CVMultiLassoPath(X, Y[:, 1:1], options; folds, nλ = 6, standardize_x, row_norm, alpha, rng = Xoshiro(7))
        @test cv.cvm ≈ scalar.cvm atol = 1e-8
        @test cv.cvse ≈ scalar.cvse atol = 1e-8
        @test cv.λmin ≈ scalar.λmin atol = 1e-8
        @test cv.λ1se ≈ scalar.λ1se atol = 1e-8
      end
    end
    @test_throws ArgumentError CVMultiLassoPath(X, Y; nfolds = 1)
    @test_throws ArgumentError CVMultiLassoPath(X, Y; folds = zeros(Int, n))
    @test_throws ArgumentError CVMultiLassoPath(X, Y; folds = [0; folds[2:end]])
    @test_throws DimensionMismatch CVMultiLassoPath(X, Y; folds = folds[2:end])
    @test_throws DimensionMismatch CVMultiLassoPath(X, Y[2:end, :])
    @test_throws ArgumentError CVMultiLassoPath(X[1:2, :], Y[1:2, :]; folds = [1, 2])
    @test_throws ArgumentError CVMultiLassoPath(X, Y; bogus = 1)
    @test_throws ArgumentError predict(CVMultiLassoPath(X, Y; folds, nλ = 3), X; λ = :other)
  end
end

end
