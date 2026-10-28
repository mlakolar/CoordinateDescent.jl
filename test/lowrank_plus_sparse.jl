module LowRankPlusSparseTests

using HighDimensionalModels
using DelimitedFiles, LinearAlgebra, Random, SparseArrays, Statistics, Test

const HDM = HighDimensionalModels
const GAP_TOL = 1e-12
uniform(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

penalty(S, w, sparsity) = sum(w[j] * (sparsity == :entry ?
  sum(abs, S[j, :]) : norm(S[j, :])) for j in axes(S, 1))
dualnorm(C, w, sparsity) = maximum((sparsity == :entry ?
  maximum(abs, C[j, :]) : norm(C[j, :])) / w[j] for j in axes(C, 1))
function objective(X, Y, L, S, b0, λr, λs, w, sparsity)
  sum(abs2, Y - X*(L+S) .- b0')/(2size(X, 1)) +
    λr*sum(svdvals(Matrix(L))) + λs*penalty(S, w, sparsity)
end

# Recompute the intersection certificate and both Fenchel equalities from
# raw data, without estimator certificate or proximal helpers.
function optimality(X, Y, s; tol = GAP_TOL)
  n = size(X, 1)
  Xc = s.hasintercept ? X .- mean(X; dims = 1) : X
  Yc = s.hasintercept ? Y .- mean(Y; dims = 1) : Y
  R = Yc - Xc*(s.L+s.S)
  C = Xc'R/n
  nr, ns = opnorm(C), dualnorm(C, s.weights, s.sparsity)
  t = min(1.0, iszero(nr) ? 1.0 : s.λ_rank/nr,
    iszero(ns) ? 1.0 : s.λ_sparse/ns)
  P = objective(X, Y, s.L, s.S, s.intercept,
    s.λ_rank, s.λ_sparse, s.weights, s.sparsity)
  D = t*dot(R, Yc)/n - t^2*sum(abs2, R)/(2n)
  gap = max(0.0, P-D)/max(P, floatmin(Float64))
  @test opnorm(t*C) <= s.λ_rank + 2e-12
  @test dualnorm(t*C, s.weights, s.sparsity) <= s.λ_sparse + 2e-12
  @test P >= D - 2e-12*max(1, P)
  @test gap <= tol + 2e-14
  @test s.gap ≈ gap atol = 2e-13 rtol = 0
  @test s.objective ≈ P rtol = 2e-13
  @test nr <= s.λ_rank + 2e-8*max(1, nr)
  @test ns <= s.λ_sparse + 2e-8*max(1, ns)
  @test s.λ_rank*sum(svdvals(s.L)) ≈ dot(C, s.L) atol = 2e-8*max(1, P) rtol = 0
  @test s.λ_sparse*penalty(s.S, s.weights, s.sparsity) ≈ dot(C, s.S) atol = 2e-8*max(1, P) rtol = 0
  if s.hasintercept
    @test norm(vec(sum(Y - predict(s, X); dims = 1))) <= 2e-11*max(1, norm(Y))
  end
  γ = inv(2opnorm(Xc)^2/n)
  F = svd(s.L + γ*C)
  PL = F.U * Diagonal(max.(F.S .- γ*s.λ_rank, 0)) * F.Vt
  PS = s.S + γ*C
  if s.sparsity == :entry
    PS = sign.(PS) .* max.(abs.(PS) .- γ*s.λ_sparse.*s.weights, 0)
  else
    for j in axes(PS, 1)
      r = norm(PS[j, :])
      PS[j, :] .*= iszero(r) ? 0 : max(0, 1-γ*s.λ_sparse*s.weights[j]/r)
    end
  end
  @test s.lowrank_residual ≈ norm(s.L-PL)/γ atol = 2e-12
  @test s.sparse_residual ≈ norm(s.S-PS)/γ atol = 2e-12
  gap
end

csv(file) = readdlm(file, ',', Float64)
function settings(file)
  rows = readdlm(file, ',', String)
  Dict(rows[i, 1] => rows[i, 2] for i in axes(rows, 1))
end

@testset "Low-rank-plus-sparse regression" begin
@testset "Clarabel fixtures (A1, A2)" begin
  root = joinpath(@__DIR__, "data", "lowrank_plus_sparse")
  for name in ("entry", "row", "entry-noint", "entry-duplicated", "row-wide")
    dir = joinpath(root, name)
    X, Y = csv(joinpath(dir, "X.csv")), csv(joinpath(dir, "Y.csv"))
    w = vec(csv(joinpath(dir, "weights.csv")))
    meta = settings(joinpath(dir, "settings.csv"))
    sparsity, intercept = Symbol(meta["sparsity"]), parse(Bool, meta["intercept"])
    problem = HDM._grid_problem(lowrank_plus_sparse, X, Y;
      sparsity, intercept, weights = w, tol = GAP_TOL)
    @test problem.λmax[1] ≈ parse(Float64, meta["lambda_rank_reference"])
    @test problem.λmax[2] ≈ parse(Float64, meta["lambda_sparse_reference"])
    for k in 1:parse(Int, meta["levels"])
      level = settings(joinpath(dir, "level$k.csv"))
      λr, λs = parse(Float64, level["lambda_rank"]), parse(Float64, level["lambda_sparse"])
      ref = parse(Float64, level["objective"])
      B = csv(joinpath(dir, "B$k.csv"))
      b0 = vec(csv(joinpath(dir, "intercept$k.csv")))
      s = lowrank_plus_sparse(X, Y, λr, λs; sparsity, intercept, weights = w, tol = GAP_TOL)
      @test s isa LowRankPlusSparseSolution{Float64}
      @test s.converged && s.gap <= GAP_TOL
      @test abs(s.objective-ref)/abs(ref) <= 1e-8
      @test maximum(abs, predict(s, X) - (X*B .+ b0')) <= 1e-5
      if name ∉ ("entry-duplicated", "row-wide")
        @test maximum(abs, s.B-B) <= 1e-5
      end
      optimality(X, Y, s)
      @test s.B == s.L+s.S
      @test coef(s) == (intercept ? [s.intercept'; s.B] : s.B)
      @test predict(s, X) + residuals(s) ≈ Y
      @test nobs(s) == size(X, 1)
      @test deviance(s) ≈ sum(abs2, Y - predict(s, X))
      @test s.intercept ≈ (intercept ? vec(mean(Y; dims = 1))-s.B'*vec(mean(X; dims = 1)) : zeros(size(Y, 2)))
      @test s.sparse_nnz == count(!iszero, s.S)
      @test s.sparse_rows == [j for j in axes(s.S, 1) if any(!iszero, s.S[j, :])]
      @test s.rank == count(>(s.rank_tol), svdvals(s.B))
      @test s.rank_tol ≈ sqrt(eps())*opnorm(s.B)
      @test 0 <= s.lowrank_rank <= min(size(s.L)...)
    end
  end
end

@testset "Identified mixed split" begin
  X = 2Matrix{Float64}(I, 4, 4)
  u, e = fill(0.5, 4), [1.0, 0, 0, 0]
  a = e-u/2
  L, S, Q = u*u', e*e', u*u' + (a*a')/2
  Y = X*(L+S+Q)
  s = lowrank_plus_sparse(X, Y, 1, 17/32; intercept = false, tol = 1e-13)
  @test s.converged && s.gap <= 1e-13
  @test s.L ≈ L atol = 1e-6 rtol = 0
  @test s.S ≈ S atol = 1e-6 rtol = 0
  @test s.objective ≈ 2.1015625 atol = 1e-12
  @test s.lowrank_rank == 1 && s.sparse_nnz == 1 && s.sparse_rows == [1]
  R = Y-X*s.B
  @test dot(R, Y)/4 - sum(abs2, R)/8 ≈ 2.1015625 atol = 1e-11
  optimality(X, Y, s; tol = 1e-13)
  for f in (dof, aic, bic, lowrank_ic)
    @test_throws MethodError f(s)
  end
  @test_throws DimensionMismatch predict(s, zeros(3, 5))
end

@testset "Strict reductions and least-squares axes (A3)" begin
  rng = Xoshiro(9101)
  A = uniform(rng, 24, 5)
  X = sqrt(24)*Matrix(qr(A .- mean(A; dims = 1)).Q)
  Y = X*uniform(rng, 5, 3) .+ [1 2 3] + 0.1uniform(rng, 24, 3)
  options = CDOptions(; opt_tol = 1e-14, gaptol = 0)
  for intercept in (false, true), sparsity in (:entry, :row), w in (ones(5), [1.0, 0.5, 2, 1.5, 0.8])
    Xc = intercept ? X .- mean(X; dims = 1) : X
    Yc = intercept ? Y .- mean(Y; dims = 1) : Y
    λr = 0.12
    λs = 1.2λr/minimum(w)
    ref = nuclear_regression(X, Y, λr; intercept, tol = 1e-12)
    @test λs > dualnorm(Xc'*(Yc-Xc*ref.B)/24, w, sparsity)
    s = lowrank_plus_sparse(X, Y, λr, λs; intercept, sparsity, weights = w, tol = 1e-13)
    @test s.converged && ref.converged
    @test s.L ≈ ref.B atol = 1e-6 rtol = 0
    @test norm(s.S) <= 1e-6
    @test predict(s, X) ≈ predict(ref, X) atol = 1e-6 rtol = 0
    @test s.objective ≈ ref.objective atol = 1e-6 rtol = 0
    λs = 0.07
    λr = 1.2λs*norm(w)*(sparsity == :entry ? sqrt(3) : 1)
    if sparsity == :entry
      refs = [lasso(X, Y[:, t], λs, w, options; intercept, rng = Xoshiro(t)) for t in 1:3]
      @test all(s -> s.converged, refs)
      B = hcat([Vector(r.x) for r in refs]...)
      b0 = [r.intercept for r in refs]
    else
      B = Xc'Yc/24
      for j in axes(B, 1)
        B[j, :] .*= max(0, 1-λs*w[j]/norm(B[j, :]))
      end
      b0 = intercept ? vec(mean(Y; dims = 1))-B'*vec(mean(X; dims = 1)) : zeros(3)
      if all(==(1), w)
        mr = multi_lasso(X, Y, λs, options; intercept, standardize_x = false)
        @test mr.converged
        @test B ≈ Matrix(mr.B) atol = 1e-9
      end
    end
    @test λr > opnorm(Xc'*(Yc-Xc*B)/24)
    s = lowrank_plus_sparse(X, Y, λr, λs; intercept, sparsity, weights = w, tol = 1e-13)
    @test s.converged
    @test norm(s.L) <= 1e-6
    @test s.S ≈ B atol = 1e-6 rtol = 0
    @test predict(s, X) ≈ X*B .+ b0' atol = 1e-6 rtol = 0
    @test s.objective ≈ objective(X, Y, zero(B), B, b0, λr, λs, w, sparsity) atol = 1e-6 rtol = 0
  end
  for design in (X, hcat(X, X[:, 1]), X[1:4, :]), intercept in (false, true)
    Z = Y[1:size(design, 1), :]
    Xc = intercept ? design .- mean(design; dims = 1) : design
    Yc = intercept ? Z .- mean(Z; dims = 1) : Z
    B = pinv(Xc)*Yc
    for (λr, λs) in ((0.0, 0.2), (0.2, 0.0), (0.0, 0.0))
      s = lowrank_plus_sparse(design, Z, λr, λs; intercept, tol = GAP_TOL)
      @test s.converged && isnan(s.gap)
      @test s.B ≈ B atol = 1e-10
      @test norm(Xc'*(Yc-Xc*s.B)) <= 1e-10
      @test s.lowrank_residual <= 1e-10 && s.sparse_residual <= 1e-10
      @test iszero(λr) ? s.L == s.B && iszero(s.S) : s.S == s.B && iszero(s.L)
    end
  end
end

@testset "Zero pair and both thresholds" begin
  X = 2Matrix{Float64}(I, 4, 4)
  Y = X*[2.0 1; 1 2; 0.5 1; 1 0.5]
  w = [1.0, 0.5, 2, 1.5]
  for sparsity in (:entry, :row)
    C = X'Y/4
    r, s = opnorm(C), dualnorm(C, w, sparsity)
    z = lowrank_plus_sparse(X, Y, 1.1r, 1.1s; sparsity, weights = w, intercept = false, tol = GAP_TOL)
    @test z.converged && z.gap <= GAP_TOL
    @test iszero(z.L) && iszero(z.S) && iszero(z.B)
    for (λr, λs) in ((1.1r, 0.2s), (0.2r, 1.1s))
      fit = lowrank_plus_sparse(X, Y, λr, λs; sparsity, weights = w, intercept = false, tol = GAP_TOL)
      @test fit.converged && fit.gap <= GAP_TOL
      @test norm(fit.B) > 0.1
    end
  end
end

@testset "Grids and isolated component warm starts (A4)" begin
  rng = Xoshiro(9102)
  X, Y = uniform(rng, 20, 4), uniform(rng, 20, 3)
  λ1, λ2 = [0.4, 0.08, 0.0], [0.3, 0.05, 0.0]
  for sparsity in (:entry, :row)
    grid = fit(PenaltyGrid, lowrank_plus_sparse, X, Y; λ1, λ2, sparsity, tol = GAP_TOL)
    @test grid.names == (:rank, :sparse)
    @test eltype(grid.solutions) === LowRankPlusSparseSolution{Float64}
    for j in CartesianIndices(grid.solutions)
      s = grid.solutions[j]
      cold = lowrank_plus_sparse(X, Y, λ1[j[1]], λ2[j[2]]; sparsity, tol = GAP_TOL)
      @test s.converged && cold.converged
      @test abs(s.objective-cold.objective)/max(cold.objective, eps()) <= 1e-8
      @test predict(s, X) ≈ predict(cold, X) atol = 1e-5 rtol = 0
    end
    @test coef(grid) == coef.(grid.solutions)
    @test predict(grid, X) == map(s -> predict(s, X), grid.solutions)
    saved = deepcopy(grid.solutions)
    for field in (:L, :S, :B, :intercept, :weights, :residuals)
      getproperty(grid.solutions[1, 1], field) .= 100
      @test getproperty(grid.solutions[2, 1], field) == getproperty(saved[2, 1], field)
    end
    automatic = PenaltyGrid(lowrank_plus_sparse, X, Y; sparsity,
      nλ = (3, 2), λminratio = (0.1, 0.2), tol = GAP_TOL)
    C = (X .- mean(X; dims = 1))'*(Y .- mean(Y; dims = 1))/20
    @test automatic.λ1 ≈ opnorm(C)*(1+sqrt(eps())) .* [1, sqrt(0.1), 0.1]
    @test automatic.λ2 ≈ dualnorm(C, ones(4), sparsity)*(1+sqrt(eps())) .* [1, 0.2]
    @test iszero(automatic.solutions[1, 1].B)
    problem = HDM._grid_problem(lowrank_plus_sparse, X, Y; sparsity, tol = GAP_TOL)
    s, warm = HDM._grid_fit(problem, 0.08, 0.05, nothing)
    snapshot, original = deepcopy(warm), deepcopy(s)
    hot, state = HDM._grid_fit(problem, 0.04, 0.03, warm)
    @test warm == snapshot
    @test hot.converged
    warm.L .= 100
    warm.S .= 200
    @test s.L == original.L && s.S == original.S && s.B == original.B
    before = copy(hot.B)
    state.L .= 100
    state.S .= 200
    @test hot.B == before
    hot.L .= -100
    @test hot.B == before
    @test hot.S != hot.L
    # The SVD convention discards null-space warm components on the axes.
    wide = hcat(X, X[:, 1])
    problem = HDM._grid_problem(lowrank_plus_sparse, wide, Y; sparsity, tol = GAP_TOL)
    ls, warm = HDM._grid_fit(problem, 0.0, 0.0, nothing)
    warm.L[1, :] .+= 1
    warm.L[end, :] .-= 1
    axis, _ = HDM._grid_fit(problem, 0.0, 0.0, warm)
    @test axis.B == ls.B
    hot, _ = HDM._grid_fit(problem, 0.08, 0.05, warm)
    cold = lowrank_plus_sparse(wide, Y, 0.08, 0.05; sparsity, tol = GAP_TOL)
    @test hot.converged && cold.converged
    @test hot.objective ≈ cold.objective rtol = 1e-8
    @test predict(hot, wide) ≈ predict(cold, wide) atol = 1e-5 rtol = 0
  end
end

@testset "CV training-only preprocessing and exclusions (A5)" begin
  rng = Xoshiro(9103)
  X = uniform(rng, 24, 4) .+ [1 2 3 4]
  Y = X*uniform(rng, 4, 2) + uniform(rng, 24, 2)
  folds = [fill(1, 5); fill(2, 8); fill(3, 11)]
  X[folds .== 1, :] .+= 2
  Y[folds .== 1, :] .+= 3
  for sparsity in (:entry, :row)
    opts = (; λ1 = [0.4, 0.1], λ2 = [0.2, 0.05], sparsity, tol = GAP_TOL)
    cv = fit(CVPenaltyGrid, lowrank_plus_sparse, X, Y; opts..., folds, threads = false, rng = Xoshiro(5))
    threaded = CVPenaltyGrid(lowrank_plus_sparse, X, Y; opts..., folds, threads = true, rng = Xoshiro(5))
    @test cv.cvm == threaded.cvm && cv.cvse == threaded.cvse
    @test coef(cv) == coef(threaded)
    @test cv.folds == folds
    @test coef(cv) == coef(cv.grid.solutions[cv.index])
    @test predict(cv, X) == predict(cv.grid.solutions[cv.index], X)
    @test cv.λmin == (cv.grid.λ1[cv.index[1]], cv.grid.λ2[cv.index[2]])
    automatic = CVPenaltyGrid(lowrank_plus_sparse, X, Y; folds, sparsity,
      nλ = (2, 2), λminratio = (0.2, 0.2), tol = GAP_TOL, threads = false)
    for result in (cv, automatic)
      errors, leaky = zeros(2, 2, 3), zeros(2, 2, 3)
      Xc, Yc = X .- mean(X; dims = 1), Y .- mean(Y; dims = 1)
      for k in 1:3
        tr, te = findall(!=(k), folds), findall(==(k), folds)
        axes = (; λ1 = result.grid.λ1, λ2 = result.grid.λ2, sparsity, tol = GAP_TOL)
        grid = PenaltyGrid(lowrank_plus_sparse, X[tr, :], Y[tr, :]; axes...)
        wrong = PenaltyGrid(lowrank_plus_sparse, Xc[tr, :], Yc[tr, :]; axes..., intercept = false)
        for j in CartesianIndices(grid.solutions)
          s = grid.solutions[j]
          @test s.converged
          @test s.intercept ≈ vec(mean(Y[tr, :]; dims = 1))-s.B'*vec(mean(X[tr, :]; dims = 1))
          errors[j, k] = mean(abs2, Y[te, :] - predict(s, X[te, :]))
          leaky[j, k] = mean(abs2, Yc[te, :] - predict(wrong.solutions[j], Xc[te, :]))
        end
      end
      counts = [5, 8, 11]
      for j in CartesianIndices(result.cvm)
        e = [errors[j, k] for k in 1:3]
        expected = dot(counts, e)/24
        se = sqrt(sum(counts .* (e .- expected).^2)/48)
        @test result.cvm[j] ≈ expected atol = 1e-10 rtol = 0
        @test result.cvse[j] ≈ se atol = 1e-10 rtol = 0
      end
      @test maximum(abs, errors-leaky) > 1e-3
    end
  end
  # The full-data gradient vanishes; each training half has a nonzero one.
  A, Z, fl = ones(6, 1), reshape([1.0, 1, 1, -1, -1, -1], 6, 1), [1, 1, 1, 2, 2, 2]
  opts = (; folds = fl, λ1 = [10.0], λ2 = [10.0, 0.01], intercept = false,
    max_iter = 1, tol = GAP_TOL, threads = false)
  excluded = CVPenaltyGrid(lowrank_plus_sparse, A, Z; opts...)
  @test all(s -> s.converged, excluded.grid.solutions)
  @test isfinite(excluded.cvm[1, 1])
  @test isnan(excluded.cvm[1, 2]) && isnan(excluded.cvse[1, 2])
  fullfailure = CVPenaltyGrid(lowrank_plus_sparse, A, ones(6, 1); opts...)
  @test !fullfailure.grid.solutions[1, 2].converged
  @test isnan(fullfailure.cvm[1, 2]) && isnan(fullfailure.cvse[1, 2])
  @test_throws ErrorException CVPenaltyGrid(lowrank_plus_sparse, A, Z;
    opts..., λ2 = [0.01])
  tie = CVPenaltyGrid(lowrank_plus_sparse, A, Z; folds = fl,
    λ1 = [2.0, 1.0], λ2 = [0.3, 0.1], threads = false)
  @test all(==(first(tie.cvm)), tie.cvm)
  @test tie.index == CartesianIndex(1, 1)
end

@testset "Boundaries and invalid arguments" begin
  rng = Xoshiro(9104)
  X, Y = uniform(rng, 20, 4), uniform(rng, 20, 3)
  for sparsity in (:entry, :row)
    for design in (zeros(20, 4), ones(20, 4)), (λr, λs) in ((0.1, 0.2), (0.0, 0.2), (0.2, 0.0), (0.0, 0.0))
      s = lowrank_plus_sparse(design, Y, λr, λs; sparsity)
      @test s.converged
      @test iszero(s.L) && iszero(s.S) && iszero(s.B)
      @test s.rank == s.lowrank_rank == s.sparse_nnz == 0
      @test s.lowrank_residual == s.sparse_residual == 0
      @test s.intercept ≈ vec(mean(Y; dims = 1))
      @test iszero(λr) || iszero(λs) ? isnan(s.gap) : s.gap <= 16eps()
    end
    zero_response = PenaltyGrid(lowrank_plus_sparse, X, zeros(20, 3); sparsity, nλ = (2, 2))
    @test iszero(zero_response.λ1) && iszero(zero_response.λ2)
    @test all(s -> s.converged && iszero(s.B), zero_response.solutions)
    f32 = lowrank_plus_sparse(Float32.(X), Float32.(Y), 0.03, 0.02; sparsity, tol = 1e-4)
    f64 = lowrank_plus_sparse(X, Y, 0.03, 0.02; sparsity, tol = GAP_TOL)
    @test f32.converged && f32.gap <= 1e-4
    @test f32.L isa Matrix{Float32} && f32.S isa Matrix{Float32} && f32.B isa Matrix{Float32}
    @test f32.objective ≈ f64.objective rtol = 1e-4
    @test predict(f32, Float32.(X)) ≈ predict(f64, X) atol = 0.01
  end
  cv32 = CVPenaltyGrid(lowrank_plus_sparse, Float32.(X), Float32.(Y);
    λ1 = [0.3, 0.1], λ2 = [0.2], folds = repeat(1:2, 10), tol = 1e-4, threads = false)
  @test cv32.cvm isa Matrix{Float32} && all(isfinite, cv32.cvm)
  @test cv32.λmin isa NTuple{2,Float32}
  w = [1.0, 0.5, 2, 1.5]
  ss = lowrank_plus_sparse(sparse(X), sparse(Y), 0.03, 0.02; weights = sparse(w), tol = GAP_TOL)
  ds = lowrank_plus_sparse(X, Y, 0.03, 0.02; weights = w, tol = GAP_TOL)
  @test ss.converged && ds.converged
  @test ss.B == ds.B && ss.L == ds.L && ss.S == ds.S && ss.weights == w
  @test lowrank_plus_sparse(ones(Int, 3, 2), ones(Int, 3, 1), 1, 1).B isa Matrix{Float64}
  short = lowrank_plus_sparse(X, Y, 0.001, 0.001; max_iter = 1, tol = 1e-14)
  @test !short.converged && short.gap > 1e-14
  @test short.objective ≈ objective(X, Y, short.L, short.S, short.intercept,
    short.λ_rank, short.λ_sparse, short.weights, short.sparsity)
  for bad in (-1.0, Inf, NaN)
    @test_throws ArgumentError lowrank_plus_sparse(X, Y, bad, 0.1)
    @test_throws ArgumentError lowrank_plus_sparse(X, Y, 0.1, bad)
  end
  for bad in (0.0, -1.0, Inf, NaN)
    @test_throws ArgumentError lowrank_plus_sparse(X, Y, 0.1, 0.1; tol = bad)
    @test_throws ArgumentError lowrank_plus_sparse(X, Y, 0.1, 0.1; weights = fill(bad, 4))
  end
  @test_throws ArgumentError lowrank_plus_sparse(X, Y, 0.1, 0.1; max_iter = 0)
  @test_throws ArgumentError lowrank_plus_sparse(X, Y, 0.1, 0.1; sparsity = :bad)
  @test_throws DimensionMismatch lowrank_plus_sparse(X, Y, 0.1, 0.1; weights = ones(3))
  @test_throws DimensionMismatch lowrank_plus_sparse(X, Y[1:3, :], 0.1, 0.1)
  @test_throws ArgumentError lowrank_plus_sparse(zeros(0, 4), zeros(0, 3), 0.1, 0.1)
  @test_throws ArgumentError lowrank_plus_sparse(zeros(20, 0), Y, 0.1, 0.1)
  @test_throws ArgumentError lowrank_plus_sparse(X, zeros(20, 0), 0.1, 0.1)
  @test_throws ArgumentError lowrank_plus_sparse(fill(NaN, 20, 4), Y, 0.1, 0.1)
  @test_throws ArgumentError lowrank_plus_sparse(X, fill(Inf, 20, 3), 0.1, 0.1)
  for bad in ([], [-1.0], [NaN], [Inf], [0.1, 0.2])
    @test_throws ArgumentError PenaltyGrid(lowrank_plus_sparse, X, Y; λ1 = bad, λ2 = [0.1])
    @test_throws ArgumentError PenaltyGrid(lowrank_plus_sparse, X, Y; λ1 = [0.1], λ2 = bad)
  end
  @test_throws ArgumentError CVPenaltyGrid(lowrank_plus_sparse, X, Y; folds = ones(Int, 20))
  @test_throws ArgumentError CVPenaltyGrid(lowrank_plus_sparse, X, Y; folds = [0; repeat(1:2, 9); 1])
  @test_throws DimensionMismatch CVPenaltyGrid(lowrank_plus_sparse, X, Y; folds = [1, 2])
end
end
end
