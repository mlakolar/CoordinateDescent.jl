module TransferTest

using HighDimensionalModels
using DelimitedFiles, LinearAlgebra, Random, SparseArrays, Statistics, Test

const HDM = HighDimensionalModels
const TIGHT = CDOptions(; opt_tol = 1e-13, gaptol = 1e-12, order = :cyclic)
const ADMM = (; abstol = 1e-12, reltol = 1e-12, max_iter = 20_000)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)
read_values(file) = Dict(row[1] => row[2] for row in eachrow(readdlm(file, ',', String)))
read_matrix(dir, file) = readdlm(joinpath(dir, file), ',', Float64)
designs(X, K) = X isa AbstractMatrix ? fill(X, K) : X
center(A, intercept) = intercept ? A .- mean(A; dims = 1) : A
rowsnorm(A) = [norm(view(A, j, :)) for j in axes(A, 1)]

# The objective, the optimality conditions of the fused problem in the original
# blocks, and the global relative duality gap, from the raw data at the
# returned blocks and intercepts, independently of the solver's coordinates.
function certificate(domains, sol; intercept)
  K = size(domains[1][2], 2)
  N = sum(size(Y, 1) for (_, Y) in domains)
  λ, a = sol.λ₀, sol.contrast_weights
  err, loss, yr, rr = 0.0, 0.0, 0.0, 0.0
  S, Sc = Matrix{Float64}[], Matrix{Float64}[]
  for (i, (X, Y)) in enumerate(domains)
    Xs, B, b = designs(X, K), sol.B[i], sol.domain_intercepts[i]
    R = hcat([Y[:, k] .- b[k] .- Xs[k] * B[:, k] for k in 1:K]...)
    loss += sum(abs2, R) / (2N * K)
    intercept && (err = max(err, maximum(abs.(mean(R; dims = 1)) ./ max.(1, sqrt.(mean(abs2, R; dims = 1))))))
    push!(S, hcat([Xs[k]' * R[:, k] for k in 1:K]...) ./ (N * K))
    # the centered scores and residuals of the dual point
    Rc = hcat([center(Y[:, k], intercept) .- center(Xs[k], intercept) * B[:, k] for k in 1:K]...)
    push!(Sc, hcat([center(Xs[k], intercept)' * Rc[:, k] for k in 1:K]...) ./ (N * K))
    yr += sum(center(Y, intercept) .* Rc); rr += sum(abs2, Rc)
  end
  function subgradient(z, Δ, c)
    nz, nd = norm(z), norm(Δ)
    iszero(nd) ? max(nz - c, 0) / max(1, c) : norm(z - c * Δ / nd) / max(1, c)
  end
  B0 = sol.B[1]
  penalty = λ * sum(rowsnorm(B0))
  for j in axes(B0, 1)
    err = max(err, subgradient(sum(Sl[j, :] for Sl in S), B0[j, :], λ))
  end
  M = maximum(norm(sum(Sl[j, :] for Sl in Sc)) for j in axes(B0, 1))
  for l in 2:length(domains)
    Δ = sol.B[l] - B0
    penalty += λ * a[l - 1] * sum(rowsnorm(Δ))
    for j in axes(B0, 1)
      err = max(err, subgradient(S[l][j, :], Δ[j, :], λ * a[l - 1]))
    end
    M = max(M, maximum(rowsnorm(Sc[l])) / a[l - 1])
  end
  P = loss + penalty
  t = iszero(M) ? 1.0 : min(1, λ / M)
  D = (t * yr - t^2 * rr / 2) / (N * K)
  (; kkt = err, objective = P, gap = (P - D) / P)
end

fitted(domains, sol) = [predict(sol, X; domain = l - 1) for (l, (X, _)) in enumerate(domains)]

function lambda_tilde_max(X, Y, W; intercept)
  K = size(Y, 2)
  Xs = designs(X, K)
  G = hcat([center(Xs[k], intercept)' * center(Y[:, k] .- Xs[k] * W[:, k], intercept) for k in 1:K]...)
  maximum(rowsnorm(G)) / (size(Y, 1) * K)
end

@testset "Transfer multitask" begin
  @testset "Clarabel references, certificates and ADMM" begin
    root = joinpath(@__DIR__, "data", "transfer")
    for name in sort(filter(d -> isdir(joinpath(root, d)), readdir(root)))
      @testset "$name" begin
        dir = joinpath(root, name)
        settings = read_values(joinpath(dir, "settings.csv"))
        L, K = parse(Int, settings["sources"]), parse(Int, settings["responses"])
        n = parse.(Int, split(settings["nobs"], ';'))
        N = sum(n)
        task = parse(Bool, settings["task_designs"])
        intercept, identified = parse(Bool, settings["intercept"]), parse(Bool, settings["identified"])
        X = [task ? [read_matrix(dir, "X$(l)_$k.csv") for k in 1:K] : read_matrix(dir, "X$l.csv") for l in 0:L]
        Y = [read_matrix(dir, "Y$l.csv") for l in 0:L]
        domains = collect(zip(X, Y))
        target, sources = domains[1], domains[2:end]
        a = vec(read_matrix(dir, "contrast_weights.csv"))
        weights = settings["contrast_weights"] == "vector" ? a : Symbol(settings["contrast_weights"])
        kw = (; contrast_weights = weights, intercept)
        top = parse(Float64, settings["lambda_max"])
        @test TransferMultitaskPath(target, sources; nλ = 1, kw...).λmax ≈ top rtol = 1e-10
        for i in 1:parse(Int, settings["levels"])
          level = read_values(joinpath(dir, "level$i.csv"))
          λ0, ref = parse(Float64, level["lambda0"]), parse(Float64, level["objective"])
          sol = transfer_multitask(target, sources, λ0, TIGHT; kw...)
          cert = certificate(domains, sol; intercept)
          @test sol.converged && sol.solver === :cd && isnothing(sol.correction)
          @test sol.contrast_weights ≈ a rtol = 1e-14
          @test abs(sol.objective - ref) <= 1e-8 * ref
          @test cert.kkt <= 1e-6
          @test sol.objective ≈ cert.objective rtol = 1e-12
          @test cert.gap <= 1e-12
          @test sol.gap ≈ cert.gap atol = 1e-13
          @test sol.W ≈ sum(n[l] / N * sol.B[l] for l in 1:L+1) atol = 1e-14
          @test sol.W == sol.fused && all(sol.contrasts[l] ≈ sol.B[l + 1] - sol.B[1] for l in 1:L)
          B = [read_matrix(dir, "B$(l)_$i.csv") for l in 0:L]
          b = [vec(read_matrix(dir, "intercept$(l)_$i.csv")) for l in 0:L]
          reference = [hcat([b[l][k] .+ designs(X[l], K)[k] * B[l][:, k] for k in 1:K]...) for l in 1:L+1]
          ours = fitted(domains, sol)
          # Fitted values are unique: an objective excess δ bounds their error by
          # √(2NKδ). One reference (curvatures, level 2, solved at 1e-11) is
          # 6e-13 above the certified optimum, which allows 2e-5; it is 1.8e-6.
          excess = max(ref - sol.objective + sol.gap * sol.objective, 0)
          @test maximum(maximum(abs, ours[l] - reference[l]) for l in 1:L+1) <= max(1e-6, sqrt(2N * K * excess))
          if name == "target-constant" && i == 2
            # The column constant in the target is fitted through the source:
            # with a₁ > 1 the target row equals the source row.
            @test any(!iszero, sol.B[1][end, :]) && all(iszero, sol.contrasts[1][end, :])
            @test maximum(abs, sol.B[1][end, :] - B[1][end, :]) <= 1e-6
          end
          if identified
            @test maximum(maximum(abs, sol.B[l] - B[l]) for l in 1:L+1) <= 1e-6
            @test maximum(abs, sol.W - read_matrix(dir, "W$i.csv")) <= 1e-6
          end
          adm = transfer_multitask(target, sources, λ0; solver = :admm, admm = ADMM, kw...)
          @test adm.converged && adm.solver === :admm
          @test abs(adm.objective - sol.objective) <= 1e-6 * sol.objective
          @test maximum(maximum(abs, f - g) for (f, g) in zip(fitted(domains, adm), ours)) <= 1e-6
          @test 0 < adm.primal_residual <= 1e-9 && 0 < adm.dual_residual <= 1e-9
          @test length(adm.penalties) == 2L + 1 && all(>(0), adm.penalties)
          @test isnan(sol.primal_residual) && isempty(sol.penalties)
          # The correction from the reference's own fused estimate.
          Wref, λt = read_matrix(dir, "W$i.csv"), parse(Float64, level["lambda_tilde"])
          corr = HDM._correction_fit(designs(X[1], K), Y[1], Wref, λt, TIGHT; intercept, rng = Xoshiro(1))
          cref = parse(Float64, level["correction_objective"])
          @test corr.converged && corr.gap <= 1e-12
          @test abs(corr.objective - cref) <= 1e-8 * cref
          @test maximum(abs, corr.D - read_matrix(dir, "D$i.csv")) <= 1e-6
          @test maximum(abs, corr.intercept - vec(read_matrix(dir, "correction_intercept$i.csv"))) <= 1e-6
          # From the package's fused estimate, within 1e-6 of the reference's.
          db = debias(sol, target, λt, TIGHT)
          @test db.correction_converged && db.converged && db.λ̃ == λt
          @test maximum(abs, db.correction - read_matrix(dir, "D$i.csv")) <= 1e-5
          @test db.W == sol.fused + db.correction && db.B == sol.B && db.objective == sol.objective
          @test predict(db, X[1]) ≈ hcat([designs(X[1], K)[k] * db.W[:, k] for k in 1:K]...) .+ db.intercept'
        end
      end
    end
  end

  rng = Xoshiro(93)
  p, K = 6, 2
  n = [24, 30, 36]
  truth = zeros(p, K); truth[1, :] .= [1.0, -0.7]; truth[2, :] .= 0.8
  shift = zeros(p, K); shift[4, :] .= 0.6
  X = [centered_rand(rng, n[i], p) .* (0.6 .+ rand(rng, p))' .+ 0.3i for i in 1:3]
  Y = [X[i] * (truth .+ (i == 3) .* shift) .+ [1.0 -0.5] .* i .+ 0.3 .* centered_rand(rng, n[i], K) for i in 1:3]
  domains = collect(zip(X, Y))
  target, sources = domains[1], domains[2:end]

  @testset "Paths, screening and λmax" begin
    grid = TransferMultitaskPath(target, sources, TIGHT; nλ = 6, λminratio = 0.05)
    unscreened = TransferMultitaskPath(target, sources, grid.λpath, TIGHT; screening = false)
    @test all(>(0), grid.kkt_checks[2:end]) && all(iszero, unscreened.kkt_checks)
    for path in (grid, unscreened), sol in path.solutions
      @test sol.converged && certificate(domains, sol; intercept = true).kkt <= 1e-6
    end
    for k in eachindex(grid.λpath)
      cold = transfer_multitask(target, sources, grid.λpath[k], TIGHT)
      @test maximum(abs, grid.solutions[k].W - unscreened.solutions[k].W) <= 1e-7
      @test maximum(abs, grid.solutions[k].W - cold.W) <= 1e-7
    end
    top = grid.λmax
    @test first(grid.λpath) ≈ top * (1 + sqrt(eps())) && last(grid.λpath) ≈ 0.05 * first(grid.λpath)
    zero = transfer_multitask(target, sources, top * (1 + 1e-8), TIGHT)
    @test all(B -> all(iszero, B), zero.B) && all(iszero, zero.W)
    @test zero.intercept ≈ vec(mean(Y[1]; dims = 1))
    @test any(B -> any(!iszero, B), transfer_multitask(target, sources, 0.99top, TIGHT).B)
    @test coef(grid, grid.λpath[3]) == coef(grid.solutions[3])
    @test predict(grid, X[1])[3] == predict(grid.solutions[3], X[1])
    @test nobs(grid) == n[1] && length(coef(grid)) == 6
    @test_throws ArgumentError coef(grid, 0.123)
    continuation = transfer_multitask(target, sources, grid.λpath[4], CDOptions(TIGHT; warm_start = false, num_steps = 3))
    @test maximum(abs, continuation.W - grid.solutions[4].W) <= 1e-7
    sp = transfer_multitask(target, [(sparse(X[2]), Y[2]), sources[2]], grid.λpath[4], TIGHT)
    @test maximum(abs, sp.W - grid.solutions[4].W) <= 1e-9
  end

  @testset "Limits" begin
    λ0 = 0.05
    # No sources: the target group lasso at K λ₀, with the same preprocessing.
    for intercept in (true, false)
      alone = transfer_multitask(target, [], λ0, TIGHT; intercept)
      ml = multi_lasso(X[1], Y[1], K * λ0, CDOptions(; opt_tol = 1e-14, order = :cyclic); intercept, standardize_x = false)
      @test maximum(abs, alone.W - Matrix(ml.B)) <= 1e-7 && maximum(abs, alone.intercept - ml.intercept) <= 1e-7
      @test alone.W == alone.B[1] && isempty(alone.contrasts) && isempty(alone.contrast_weights)
    end
    # Large contrast weights: common slopes with domain intercepts, i.e. pooling
    # the data centered within each domain.
    pooled = transfer_multitask(target, sources, λ0, TIGHT; contrast_weights = [1e4, 1e4])
    @test all(C -> all(iszero, C), pooled.contrasts)
    @test pooled.W ≈ pooled.B[1] atol = 1e-15
    ml = multi_lasso(vcat(center.(X, true)...), vcat(center.(Y, true)...), K * λ0,
                     CDOptions(; opt_tol = 1e-14, order = :cyclic); intercept = false, standardize_x = false)
    @test maximum(abs, pooled.W - Matrix(ml.B)) <= 1e-7
    for l in 1:3
      @test pooled.domain_intercepts[l] ≈ vec(mean(Y[l]; dims = 1) - mean(X[l]; dims = 1) * pooled.W) atol = 1e-12
    end
    # A correction at or above λ̃max is zero.
    fusedfit = transfer_multitask(target, sources, λ0, TIGHT)
    tmax = lambda_tilde_max(X[1], Y[1], fusedfit.fused; intercept = true)
    none = debias(fusedfit, target, tmax * (1 + 1e-8), TIGHT)
    @test all(iszero, none.correction) && none.W == none.fused && none.intercept ≈ fusedfit.intercept
    some = debias(fusedfit, target, 0.9tmax, TIGHT)
    @test any(!iszero, some.correction) && some.correction_converged
    @test some.correction_gap <= 1e-12
    @test debias(some, target, tmax * (1 + 1e-8), TIGHT).W == fusedfit.fused
    ml = multi_lasso(X[1], Y[1] - X[1] * fusedfit.fused, K * 0.9tmax, CDOptions(; opt_tol = 1e-14, order = :cyclic);
                     standardize_x = false)
    @test maximum(abs, some.correction - Matrix(ml.B)) <= 1e-7
    # Task-specific designs equal to the shared ones give the shared fit.
    tasks = transfer_multitask((fill(X[1], K), Y[1]), [([X[2], copy(X[2])], Y[2]), sources[2]], λ0, TIGHT)
    @test maximum(abs, tasks.W - fusedfit.W) <= 1e-9
    @test predict(tasks, [X[1], X[1]]) ≈ predict(fusedfit, X[1])
  end

  @testset "Budgets and StatsAPI" begin
    @test !any(s -> s.converged, TransferMultitaskPath(target, sources, [0.2, 0.1], CDOptions(; max_iter = 0)).solutions)
    @test !transfer_multitask(target, sources, 0.02, CDOptions(; max_iter = 1, opt_tol = 0, gaptol = 1e-12)).converged
    short = transfer_multitask(target, sources, 0.02; solver = :admm, admm = (; max_iter = 3))
    @test !short.converged && isfinite(short.primal_residual) && isfinite(short.dual_residual)
    none = transfer_multitask(target, sources, 0.02; solver = :admm, admm = (; max_iter = 0))
    @test !none.converged && isnan(none.primal_residual)
    sol = transfer_multitask(target, sources, 0.05)
    @test !debias(sol, target, 0.001, CDOptions(; max_iter = 0)).correction_converged
    @test size(coef(sol)) == (p + 1, K) && coef(sol)[1, :] == sol.intercept && coef(sol)[2:end, :] == sol.W
    @test coef(sol; domain = 2)[2:end, :] == sol.B[3] && nobs(sol) == n[1]
    @test predict(sol, X[3]; domain = 2) ≈ X[3] * sol.B[3] .+ sol.domain_intercepts[3]'
    @test predict(sol, X[1]; domain = 0) ≈ X[1] * sol.B[1] .+ sol.domain_intercepts[1]'
    @test isnan(sol.λ̃) && isnan(sol.correction_gap) && !sol.correction_converged
    @test fit(TransferMultitaskSolution, target, sources, 0.05, TIGHT).W == transfer_multitask(target, sources, 0.05, TIGHT).W
    @test fit(TransferMultitaskPath, target, sources, [0.1], TIGHT).λpath == [0.1]
    f32 = transfer_multitask((Float32.(X[1]), Float32.(Y[1])), [(Float32.(X[2]), Float32.(Y[2]))], 0.05f0)
    @test eltype(f32.W) === Float32 && f32.converged
  end

  @testset "CV against a manual computation, and threads" begin
    folds = repeat(1:3, 8)
    families = (:sqrt, :linear)
    kw = (; folds, families, nλ = 4, λminratio = 0.1, nλ̃ = 4)
    cv = CVTransferMultitaskPath(target, sources, TIGHT; kw..., threads = false, rng = Xoshiro(2))
    parallel = CVTransferMultitaskPath(target, sources, TIGHT; kw..., threads = true, rng = Xoshiro(2))
    @test cv.cvm == parallel.cvm && cv.cvm_correction == parallel.cvm_correction
    @test cv.selected == parallel.selected && cv.λ̃ == parallel.λ̃
    m = [count(==(k), folds) for k in 1:3]
    sse = zeros(2, 3, 4)
    foldfits = Matrix{Any}(undef, 2, 3)
    for i in 1:2, k in 1:3
      train, test = findall(!=(k), folds), findall(==(k), folds)
      fp = TransferMultitaskPath((X[1][train, :], Y[1][train, :]), sources, cv.paths[i].λpath, TIGHT;
                                 contrast_weights = families[i])
      Ntrain = sum(n) - m[k]
      @test fp.contrast_weights ≈ (i == 1 ? 8 .* sqrt.(n[2:3] ./ Ntrain) : 8 .* n[2:3] ./ Ntrain)
      @test all(s -> s.converged, fp.solutions)
      foldfits[i, k] = fp
      for j in 1:4
        sse[i, k, j] = sum(abs2, Y[1][test, :] - predict(fp.solutions[j], X[1][test, :]))
      end
    end
    manual = dropdims(sum(sse; dims = 2); dims = 2) ./ (K * n[1])
    @test cv.cvm ≈ manual atol = 1e-10
    @test cv.selected == argmin(manual) && cv.λ₀ == cv.paths[cv.selected[1]].λpath[cv.selected[2]]
    @test cv.family === families[cv.selected[1]] && cv.fused === cv.paths[cv.selected[1]].solutions[cv.selected[2]]
    i, j = Tuple(cv.selected)
    @test first(cv.λ̃path) / (1 + sqrt(eps())) ≈ lambda_tilde_max(X[1], Y[1], cv.fused.fused; intercept = true) rtol = 1e-12
    sse2 = zeros(3, 4)
    for k in 1:3
      train, test = findall(!=(k), folds), findall(==(k), folds)
      for (l, λt) in enumerate(cv.λ̃path)
        db = debias(foldfits[i, k].solutions[j], (X[1][train, :], Y[1][train, :]), λt, TIGHT)
        sse2[k, l] = sum(abs2, Y[1][test, :] - predict(db, X[1][test, :]))
      end
    end
    manual2 = vec(sum(sse2; dims = 1)) ./ (K * n[1])
    @test cv.cvm_correction ≈ manual2 atol = 1e-10
    @test cv.λ̃ == cv.λ̃path[argmin(manual2)]
    full = debias(cv.fused, target, cv.λ̃, TIGHT)
    @test cv.solution.correction_converged && maximum(abs, cv.solution.W - full.W) <= 1e-7
    @test coef(cv) == coef(cv.solution) && predict(cv, X[1]) == predict(cv.solution, X[1]) && nobs(cv) == n[1]
    @test cv.folds == folds && size(cv.cvm) == (2, 4) && all(isfinite, cv.cvse)
    single = CVTransferMultitaskPath(target, sources, TIGHT; folds, families = :sqrt, nλ = 3, λ̃path = [0.1, 0.01])
    @test single.families == [:sqrt] && single.λ̃path == [0.1, 0.01]
    @test_throws ArgumentError CVTransferMultitaskPath(target, sources, CDOptions(; max_iter = 0); folds, nλ = 2)
  end

  @testset "Input checks" begin
    for λ0 in (-0.1, Inf, NaN)
      @test_throws ArgumentError transfer_multitask(target, sources, λ0)
    end
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; contrast_weights = :other)
    @test_throws DimensionMismatch transfer_multitask(target, sources, 0.1; contrast_weights = [1.0])
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; contrast_weights = [1.0, 0.0])
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; solver = :other)
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; solver = :admm, admm = (; rho = -1))
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; solver = :admm, admm = (; other = 1))
    @test_throws ArgumentError transfer_multitask(target, sources, 0.1; warm_start = true)
    @test_throws DimensionMismatch transfer_multitask(target, [(X[2], Y[2][:, 1:1])], 0.1)
    @test_throws DimensionMismatch transfer_multitask(target, [(X[2][:, 1:3], Y[2])], 0.1)
    @test_throws DimensionMismatch transfer_multitask(target, [(X[2][1:5, :], Y[2])], 0.1)
    @test_throws DimensionMismatch transfer_multitask(target, [([X[2]], Y[2])], 0.1)
    @test_throws DimensionMismatch transfer_multitask(target, [([X[2], X[2][1:5, :]], Y[2])], 0.1)
    @test_throws ArgumentError transfer_multitask(target, [(X[2], fill(NaN, n[2], K))], 0.1)
    @test_throws ArgumentError transfer_multitask(target, [X[2]], 0.1)
    @test_throws ArgumentError transfer_multitask((X[1][1:1, :], Y[1][1:1, :]), sources, 0.1)
    sol = transfer_multitask(target, sources, 0.1)
    @test_throws ArgumentError debias(sol, target, -1.0)
    @test_throws DimensionMismatch debias(sol, (X[1][1:10, :], Y[1][1:10, :]), 0.1)
    @test_throws DimensionMismatch predict(sol, X[1][:, 1:3])
    @test_throws ArgumentError predict(sol, X[1]; domain = 3)
    @test_throws DimensionMismatch predict(sol, [X[1]])
    @test_throws ArgumentError TransferMultitaskPath(target, sources; nλ = 0)
    @test_throws ArgumentError TransferMultitaskPath(target, sources; λminratio = 2)
    @test_throws ArgumentError TransferMultitaskPath(target, sources, [0.1, 0.2])
    @test_throws ArgumentError TransferMultitaskPath(target, sources, [-0.1])
    @test_throws ArgumentError CVTransferMultitaskPath(target, sources; families = ())
    @test_throws ArgumentError CVTransferMultitaskPath(target, sources; nfolds = 1)
    @test_throws DimensionMismatch CVTransferMultitaskPath(target, sources; folds = ones(Int, 3))
    @test_throws ArgumentError CVTransferMultitaskPath(target, sources; folds = [fill(1, 23); 2])
  end
end

end
