module SparseNuclearTests

using HighDimensionalModels
using DelimitedFiles, LinearAlgebra, Random, SparseArrays, Statistics, Test
import ProximalBase

const HDM = HighDimensionalModels
const GAP_TOL = 1e-10
uniform(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

function penalty(B, w, sparsity)
  sparsity == :entry ? sum(w .* vec(sum(abs.(B); dims = 2))) :
    sum(w[j] * norm(B[j, :]) for j in axes(B, 1))
end
function objective(X, Y, B, b0, λr, λs, w, sparsity)
  sum(abs2, Y - X*B .- b0')/(2size(X, 1)) + λr*sum(svdvals(Matrix(B))) + λs*penalty(B, w, sparsity)
end
function dualnorm(V, w, sparsity)
  sparsity == :entry ? maximum(abs.(V) ./ w) :
    maximum(norm(V[j, :])/w[j] for j in axes(V, 1))
end

# Independent scaling of the returned residual dual, using only raw data and
# the copied solver dual. No estimator certificate helper is called here.
function certificate(X, Y, s, y)
  n, p = size(X)
  Xc = s.hasintercept ? X .- mean(X; dims = 1) : X
  Yc = s.hasintercept ? Y .- mean(Y; dims = 1) : Y
  R = Yc - Xc*s.B
  C = Xc'R/n
  λr, λs, w = s.λ_rank, s.λ_sparse, s.weights
  U = λs == 0 ? C : λr == 0 ? zero(C) : reshape(copy(y), size(C))
  V = C - U
  t = 1.0
  if λr > 0 && opnorm(U) > 0
    t = min(t, λr/opnorm(U))
  end
  if λs > 0 && dualnorm(V, w, s.sparsity) > 0
    t = min(t, λs/dualnorm(V, w, s.sparsity))
  end
  P = objective(X, Y, s.B, s.intercept, λr, λs, w, s.sparsity)
  D = dot(t .* R, Yc)/n - sum(abs2, t .* R)/(2n)
  @test opnorm(t .* U) <= λr + 2e-12
  @test dualnorm(t .* V, w, s.sparsity) <= λs + 2e-12
  @test P >= D - 2e-12*max(1, P)
  max(0.0, P-D)/max(P, floatmin(Float64))
end

function project_sparse(V, λ, w, sparsity)
  if sparsity == :entry
    return clamp.(V, -λ .* w, λ .* w)
  end
  Z = copy(V)
  for j in axes(Z, 1)
    nr = norm(Z[j, :])
    nr == 0 || (Z[j, :] .*= min(1, λ*w[j]/nr))
  end
  Z
end

# Alternating projections onto the subgradient faces at B, independently of
# the solver dual: at most 5000 projections, feasibility tolerance 1e-8.
# Using faces avoids tangential convergence of projections onto whole balls.
# Complementarity is bounded by the gap plus the remaining feasibility error.
function optimality(X, Y, s)
  n = size(X, 1)
  Xc = s.hasintercept ? X .- mean(X; dims = 1) : X
  R = Y - predict(s, X)
  C = Xc'R/n
  F = svd(s.B)
  # A fit below the feasibility tolerance has unresolved subgradient faces.
  r = norm(s.B) <= 1e-8 ? 0 : count(>(sqrt(eps())*first(F.S)), F.S)
  L, Q = F.U[:, 1:r], F.V[:, 1:r]
  base = s.λ_rank * L * Q'
  U, V = zero(C), zero(C)
  for _ in 1:5000
    W = C - V - base
    W -= L*(L'W)
    W -= (W*Q)*Q'
    H = svd(W)
    U = base + H.U * Diagonal(min.(H.S, s.λ_rank)) * H.Vt
    V = project_sparse(C - U, s.λ_sparse, s.weights, s.sparsity)
    if r > 0 && s.sparsity == :entry
      for j in axes(V, 1), t in axes(V, 2)
        iszero(s.B[j, t]) || (V[j, t] = s.λ_sparse*s.weights[j]*sign(s.B[j, t]))
      end
    elseif r > 0
      for j in axes(V, 1)
        nb = norm(s.B[j, :])
        nb == 0 || (V[j, :] .= (s.λ_sparse*s.weights[j]/nb) .* s.B[j, :])
      end
    end
    norm(C-U-V) <= 1e-8*max(1, norm(C)) && break
  end
  error = norm(C-U-V)
  @test error <= 1e-8*max(1, norm(C))
  @test opnorm(U) <= s.λ_rank + 2e-12
  @test dualnorm(V, s.weights, s.sparsity) <= s.λ_sparse + 2e-12
  budget = s.gap*max(s.objective, floatmin(Float64)) + error*norm(s.B) + 5e-12*max(1, s.objective)
  cr = s.λ_rank*sum(svdvals(s.B)) - dot(U, s.B)
  cs = s.λ_sparse*penalty(s.B, s.weights, s.sparsity) - dot(V, s.B)
  @test -5e-12 <= cr <= budget
  @test -5e-12 <= cs <= budget
end

csv(path) = readdlm(path, ',', Float64)
function settings(path)
  rows = readdlm(path, ',', String)
  Dict(rows[i, 1] => rows[i, 2] for i in axes(rows, 1))
end

@testset "Sparse nuclear regression" begin
@testset "Clarabel fixtures (A1, A2)" begin
  root = joinpath(@__DIR__, "data", "lowrank_sparse")
  for name in ("entry", "row", "entry-wide", "row-noint")
    dir = joinpath(root, name)
    X, Y, w = csv(joinpath(dir, "X.csv")), csv(joinpath(dir, "Y.csv")), vec(csv(joinpath(dir, "weights.csv")))
    meta = settings(joinpath(dir, "settings.csv"))
    sparsity, intercept = Symbol(meta["sparsity"]), parse(Bool, meta["intercept"])
    problem = HDM._grid_problem(sparse_nuclear_regression, X, Y;
      sparsity, intercept, weights = w, tol = GAP_TOL)
    @test problem.λmax[1] ≈ parse(Float64, meta["lambda_rank_max"])
    @test problem.λmax[2] ≈ parse(Float64, meta["lambda_sparse_max"])
    for k in 1:parse(Int, meta["levels"])
      level = settings(joinpath(dir, "level$k.csv"))
      λr, λs = parse(Float64, level["lambda_rank"]), parse(Float64, level["lambda_sparse"])
      ref = parse(Float64, level["objective"])
      B, b0 = csv(joinpath(dir, "B$k.csv")), vec(csv(joinpath(dir, "intercept$k.csv")))
      s, warm = HDM._grid_fit(problem, λr, λs, nothing)
      @test s isa SparseNuclearSolution{Float64}
      @test s.converged && s.gap <= GAP_TOL
      @test abs(s.objective-ref)/abs(ref) <= 1e-8
      @test maximum(abs, predict(s, X) - (X*B .+ b0')) <= 1e-5
      if name != "entry-wide"
        @test maximum(abs, s.B - B) <= 1e-5
      end
      @test s.objective ≈ objective(X, Y, s.B, s.intercept, λr, λs, w, sparsity) rtol = 2e-14
      independent_gap = certificate(X, Y, s, warm.y)
      @test independent_gap <= GAP_TOL
      @test independent_gap ≈ s.gap atol = 2e-13 rtol = 0
      optimality(X, Y, s)
      @test coef(s) == (intercept ? [s.intercept'; s.B] : s.B)
      @test predict(s, X) + residuals(s) ≈ Y
      @test nobs(s) == size(X, 1)
      @test deviance(s) ≈ sum(abs2, Y - predict(s, X))
      @test s.intercept ≈ (intercept ? vec(mean(Y; dims = 1)) - s.B'*vec(mean(X; dims = 1)) : zeros(size(Y, 2)))
      @test s.activerows == [j for j in axes(s.B, 1) if any(!iszero, s.B[j, :])]
      @test s.nnz == count(!iszero, s.B)
      @test s.rank == count(>(s.rank_tol), svdvals(s.B))
      @test s.rank_tol >= sqrt(eps(Float64))*opnorm(s.B)
      @test all(isfinite, (s.r_norm, s.s_norm, s.eps_pri, s.eps_dual))
      if name == "entry" && k == 1
        @test λr < problem.λmax[1] && λs < problem.λmax[2]
        @test norm(s.B) <= 1e-8
      end
    end
  end
end

@testset "Sparse nuclear adapters and analytic solution" begin
  rng = Xoshiro(9201)
  for (n, p) in ((12, 4), (4, 7))
    X, Y = uniform(rng, n, p), uniform(rng, n, 3)
    problem = HDM._grid_problem(sparse_nuclear_regression, X, Y)
    x, direction = rand(rng, p*3), uniform(rng, p*3)
    L = ProximalBase.lipschitz(problem.loss)
    @test L == problem.loss.setup.L
    curvature = problem.loss(x + direction) + problem.loss(x - direction) - 2problem.loss(x)
    @test 0 <= curvature <= L*sum(abs2, direction)
    G = similar(x)
    ProximalBase.gradient!(G, problem.loss, x)
    Xc, Yc = X .- mean(X; dims = 1), Y .- mean(Y; dims = 1)
    @test reshape(G, p, 3) ≈ Xc'*(Xc*reshape(x, p, 3)-Yc)/n
    @test ProximalBase.value_and_gradient!(G, problem.loss, x) ≈ sum(abs2, Yc-Xc*reshape(x, p, 3))/(2n)
    fd = (problem.loss(x + 1e-6direction) - problem.loss(x - 1e-6direction))/2e-6
    @test dot(G, direction) ≈ fd atol = 1e-8
  end
  for T in (Float32, Float64)
    A = T[3 1; -1 2; 0 0]
    h = HDM._SparseNuclearShape(ProximalBase.NuclearNorm(T(0.7)), zeros(T, 3, 2))
    F = svd(A)
    expected = F.U * Diagonal(min.(F.S, T(0.7))) * F.Vt
    x = vec(copy(A))
    @test h(x) ≈ T(0.7)*sum(F.S)
    hc = ProximalBase.Conjugate(h)
    ProximalBase.prox!(x, hc, x, T(0.3))
    @test reshape(x, 3, 2) ≈ expected atol = 20eps(T)
  end
  X = 2Matrix{Float64}(I, 4, 4)
  c = [3.0, 0.7, 0.2, 0]
  Y = X * Diagonal(c)
  s = sparse_nuclear_regression(X, Y, 0.3, 0.2; intercept = false, tol = 1e-12)
  @test s.converged && s.gap <= 1e-12
  @test s.B ≈ Diagonal(max.(c .- 0.5, 0)) atol = 1e-10
  @test s.rank == 2
  @test_throws MethodError dof(s)
  @test_throws MethodError aic(s)
  @test_throws MethodError bic(s)
  @test_throws MethodError lowrank_ic(s)
  @test_throws DimensionMismatch predict(s, zeros(3, 5))
end

@testset "Zero-penalty reductions (A3)" begin
  rng = Xoshiro(9202)
  X = uniform(rng, 32, 5) .+ [1 2 0 -1 3]
  Y = X*uniform(rng, 5, 3) .+ [1 2 3] + 0.1uniform(rng, 32, 3)
  opts = CDOptions(; opt_tol = 1e-14, gaptol = 0)
  for intercept in (false, true)
    s = sparse_nuclear_regression(X, Y, 0.08, 0; intercept, tol = 1e-12)
    ref = nuclear_regression(X, Y, 0.08; intercept, tol = 1e-12)
    @test s.converged && ref.converged
    @test s.B ≈ ref.B atol = 1e-6 rtol = 0
    @test predict(s, X) ≈ predict(ref, X) atol = 1e-6 rtol = 0
    @test s.objective ≈ ref.objective atol = 1e-6 rtol = 0
    for w in (ones(5), [1.0, 0.5, 2, 1.5, 0.8])
      se = sparse_nuclear_regression(X, Y, 0, 0.07; intercept, weights = w, tol = 1e-12)
      refs = [lasso(X, Y[:, t], 0.07, w, opts; intercept, rng = Xoshiro(t)) for t in 1:3]
      B = hcat([Vector(s.x) for s in refs]...)
      predictions = hcat([predict(s, X) for s in refs]...)
      @test se.converged && all(s -> s.converged, refs)
      @test se.B ≈ B atol = 1e-6 rtol = 0
      @test predict(se, X) ≈ predictions atol = 1e-6 rtol = 0
      @test se.objective ≈ objective(X, Y, B, [s.intercept for s in refs], 0, 0.07, w, :entry) atol = 1e-6 rtol = 0
      sr = sparse_nuclear_regression(X, Y, 0, 0.07; sparsity = :row, intercept, weights = w, tol = 1e-12)
      mr = w == ones(5) ? multi_lasso(X, Y, 0.07, opts; intercept, standardize_x = false) :
        multitask_lasso([X for _ in 1:3], [Y[:, t] for t in 1:3], 0.07, opts;
          intercept, standardize_x = false, penalty_factors = w, rng = Xoshiro(1))
      if !sr.converged || !mr.converged
        @info "Row reduction termination" intercept weights = w gap = sr.gap sparse_iterations = sr.iterations sparse_converged = sr.converged baseline_converged = mr.converged baseline_iterations = mr.iterations
      end
      @test sr.converged
      @test mr.converged
      @test sr.B ≈ Matrix(mr.B) atol = 1e-6 rtol = 0
      @test predict(sr, X) ≈ X*mr.B .+ mr.intercept' atol = 1e-6 rtol = 0
      @test sr.objective ≈ objective(X, Y, mr.B, mr.intercept, 0, 0.07, w, :row) atol = 1e-6 rtol = 0
    end
    ls = sparse_nuclear_regression(X, Y, 0, 0; intercept, tol = 1e-12)
    Xc = intercept ? X .- mean(X; dims = 1) : X
    Yc = intercept ? Y .- mean(Y; dims = 1) : Y
    @test ls.converged && isnan(ls.gap)
    @test ls.B ≈ Xc \ Yc atol = 1e-6 rtol = 0
    @test ls.r_norm <= ls.eps_pri && ls.s_norm <= ls.eps_dual
  end
end

@testset "Penalty grids and independent warm states (A4)" begin
  rng = Xoshiro(9203)
  A = uniform(rng, 24, 5)
  X = Matrix(qr(A .- mean(A; dims = 1)).Q) * sqrt(24)
  Y = X*uniform(rng, 5, 3) + 0.1uniform(rng, 24, 3)
  λ1, λ2 = [0.4, 0.1, 0.0], [0.2, 0.05, 0.0]
  for sparsity in (:entry, :row)
    grid = fit(PenaltyGrid, sparse_nuclear_regression, X, Y; λ1, λ2, sparsity, tol = 1e-12)
    @test grid.names == (:rank, :sparse)
    @test eltype(grid.solutions) === SparseNuclearSolution{Float64}
    for j in CartesianIndices(grid.solutions)
      s = grid.solutions[j]
      cold = sparse_nuclear_regression(X, Y, λ1[j[1]], λ2[j[2]]; sparsity, tol = 1e-12)
      @test s.converged && cold.converged
      @test abs(s.objective-cold.objective)/max(cold.objective, eps()) <= 1e-8
      @test predict(s, X) ≈ predict(cold, X) atol = 1e-5 rtol = 0
    end
    @test coef(grid) == coef.(grid.solutions)
    @test predict(grid, X) == map(s -> predict(s, X), grid.solutions)
    original = deepcopy(grid.solutions[2, 1])
    grid.solutions[1, 1].B .= 100
    grid.solutions[1, 1].intercept .= 100
    grid.solutions[1, 1].weights .= 100
    grid.solutions[1, 1].residuals .= 100
    @test grid.solutions[2, 1].B == original.B
    @test grid.solutions[2, 1].intercept == original.intercept
    @test grid.solutions[2, 1].weights == original.weights
    @test grid.solutions[2, 1].residuals == original.residuals
    automatic = PenaltyGrid(sparse_nuclear_regression, X, Y; sparsity, nλ = (3, 2), λminratio = (0.1, 0.2))
    C = X'*(Y .- mean(Y; dims = 1))/24
    @test automatic.λ1 ≈ opnorm(C)*(1+sqrt(eps())) .* [1, sqrt(0.1), 0.1]
    @test automatic.λ2 ≈ dualnorm(C, ones(5), sparsity)*(1+sqrt(eps())) .* [1, 0.2]
    @test iszero(automatic.solutions[1, 1].B)
  end
  # With zero penalties, an initial null-space component survives. A later
  # penalized fit must still agree with a cold fit in objective and predictions.
  Xw = hcat(X, X[:, 1])
  problem = HDM._grid_problem(sparse_nuclear_regression, Xw, Y; tol = 1e-12)
  cold, warm = HDM._grid_fit(problem, 0.0, 0.0, nothing)
  N = zeros(6, 3); N[1, :] .= 1; N[end, :] .= -1
  warm.x .+= vec(N)
  saved = deepcopy(warm)
  hot, state = HDM._grid_fit(problem, 0.0, 0.0, warm)
  @test hot.converged
  @test hot.B ≈ cold.B + N atol = 1e-9
  @test warm == saved
  hot2, _ = HDM._grid_fit(problem, 0.1, 0.05, state)
  cold2 = sparse_nuclear_regression(Xw, Y, 0.1, 0.05; tol = 1e-12)
  @test hot2.converged && cold2.converged
  @test abs(hot2.objective-cold2.objective)/cold2.objective <= 1e-8
  @test predict(hot2, Xw) ≈ predict(cold2, Xw) atol = 1e-5
  state.x .= 0
  state.y .= 0
  @test hot.B ≈ cold.B + N atol = 1e-9
end

@testset "CV training folds, weights and exclusions (A5)" begin
  rng = Xoshiro(9204)
  X = uniform(rng, 24, 4) .+ [1 2 3 4]
  Y = X*uniform(rng, 4, 2) + uniform(rng, 24, 2)
  folds = [fill(1, 5); fill(2, 8); fill(3, 11)]
  Y[folds .== 1, :] .+= 3
  opts = (; λ1 = [0.4, 0.1], λ2 = [0.2, 0.05], tol = 1e-12)
  cv = fit(CVPenaltyGrid, sparse_nuclear_regression, X, Y; opts..., folds, threads = false, rng = Xoshiro(5))
  threaded = CVPenaltyGrid(sparse_nuclear_regression, X, Y; opts..., folds, threads = true, rng = Xoshiro(5))
  @test cv.cvm == threaded.cvm
  @test cv.cvse == threaded.cvse
  @test coef(cv) == coef(threaded)
  @test cv.folds == folds
  @test coef(cv) == coef(cv.grid.solutions[cv.index])
  @test predict(cv, X) == predict(cv.grid.solutions[cv.index], X)
  @test cv.index == argmin(cv.cvm)
  @test cv.λmin == (cv.grid.λ1[cv.index[1]], cv.grid.λ2[cv.index[2]])
  function manual(cv)
    errors = zeros(2, 2, 3)
    for k in 1:3
      tr, te = findall(!=(k), folds), findall(==(k), folds)
      grid = PenaltyGrid(sparse_nuclear_regression, X[tr, :], Y[tr, :];
        λ1 = cv.grid.λ1, λ2 = cv.grid.λ2, tol = 1e-12)
      for j in CartesianIndices(grid.solutions)
        @test grid.solutions[j].converged
        errors[j, k] = mean(abs2, Y[te, :] - predict(grid.solutions[j], X[te, :]))
      end
    end
    counts = [5, 8, 11]
    for j in CartesianIndices(cv.cvm)
      e = [errors[j, k] for k in 1:3]
      expected = sum(counts .* e)/24
      se = sqrt(sum(counts .* (e .- expected).^2)/48)
      @test cv.cvm[j] ≈ expected atol = 1e-10 rtol = 0
      @test cv.cvse[j] ≈ se atol = 1e-10 rtol = 0
    end
  end
  manual(cv)
  automatic = CVPenaltyGrid(sparse_nuclear_regression, X, Y; folds,
    nλ = (2, 2), λminratio = (0.2, 0.2), tol = 1e-12, threads = false)
  manual(automatic)
  # The full gradient vanishes, but each training half has a nonzero gradient.
  # Thus only the training fits at the lower pair fail in a single iteration.
  A, Z, fl = ones(6, 1), reshape([1.0, 1, 1, -1, -1, -1], 6, 1), [1, 1, 1, 2, 2, 2]
  excluded = CVPenaltyGrid(sparse_nuclear_regression, A, Z; folds = fl,
    λ1 = [0.0], λ2 = [10.0, 0.01], intercept = false, max_iter = 1, tol = 1e-12, threads = false)
  @test all(s -> s.converged, excluded.grid.solutions)
  @test isfinite(excluded.cvm[1, 1])
  @test isnan(excluded.cvm[1, 2]) && isnan(excluded.cvse[1, 2])
  @test excluded.index == CartesianIndex(1, 1)
  fullfailure = CVPenaltyGrid(sparse_nuclear_regression, A, ones(6, 1); folds = fl,
    λ1 = [0.0], λ2 = [10.0, 0.01], intercept = false, max_iter = 1, tol = 1e-12, threads = false)
  @test !fullfailure.grid.solutions[1, 2].converged
  @test isnan(fullfailure.cvm[1, 2]) && isnan(fullfailure.cvse[1, 2])
  @test_throws ErrorException CVPenaltyGrid(sparse_nuclear_regression, A, Z; folds = fl,
    λ1 = [0.0], λ2 = [0.01], intercept = false, max_iter = 1, tol = 1e-12, threads = false)
  tie = CVPenaltyGrid(sparse_nuclear_regression, A, Z; folds = fl,
    λ1 = [2.0, 1.0], λ2 = [0.3, 0.1], threads = false)
  @test all(==(first(tie.cvm)), tie.cvm)
  @test tie.index == CartesianIndex(1, 1)
end

@testset "Sparse nuclear boundaries and invalid inputs" begin
  rng = Xoshiro(9205)
  X, Y = uniform(rng, 24, 4), uniform(rng, 24, 3)
  for sparsity in (:entry, :row)
    Xc, Yc = X .- mean(X; dims = 1), Y .- mean(Y; dims = 1)
    C = Xc'Yc/24
    for (λr, λs) in ((1.01opnorm(C), 0.0), (0.0, 1.01dualnorm(C, ones(4), sparsity)))
      s = sparse_nuclear_regression(X, Y, λr, λs; sparsity, tol = 1e-12)
      @test s.converged && s.gap <= 1e-12
      @test norm(s.B) <= 1e-8
    end
    z = PenaltyGrid(sparse_nuclear_regression, X, zeros(24, 3); sparsity, nλ = (2, 2))
    @test iszero(z.λ1) && iszero(z.λ2)
    @test all(s -> s.converged && iszero(s.B), z.solutions)
    f32 = sparse_nuclear_regression(Float32.(X), Float32.(Y), 0.03, 0.02; sparsity, tol = 1e-4)
    f64 = sparse_nuclear_regression(X, Y, 0.03, 0.02; sparsity, tol = 1e-10)
    @test f32.converged && f32.gap <= 1e-4
    @test f32.B isa Matrix{Float32}
    @test f32.objective ≈ f64.objective rtol = 1e-4
    @test predict(f32, Float32.(X)) ≈ predict(f64, X) atol = 0.01
    rd = sparse_nuclear_regression(hcat(X, X[:, 1]), Y, 0.03, 0.02; sparsity, tol = 1e-10)
    @test rd.converged && rd.gap <= 1e-10
    cs = sparse_nuclear_regression(ones(24, 4), Y, 0.03, 0.02; sparsity)
    @test cs.converged && cs.iterations == 0 && cs.gap == 0
    @test iszero(cs.B) && cs.rank == cs.nnz == 0
    @test cs.intercept ≈ vec(mean(Y; dims = 1))
  end
  cv32 = CVPenaltyGrid(sparse_nuclear_regression, Float32.(X), Float32.(Y);
    λ1 = [0.3, 0.1], λ2 = [0.2], folds = repeat(1:3, 8), tol = 1e-4, threads = false)
  @test cv32.cvm isa Matrix{Float32}
  @test all(isfinite, cv32.cvm)
  @test cv32.λmin isa NTuple{2,Float32}
  w = [1.0, 0.5, 2, 1.5]
  sparsefit = sparse_nuclear_regression(sparse(X), sparse(Y), 0.03, 0.02;
    weights = sparse(w), tol = 1e-10)
  densefit = sparse_nuclear_regression(X, Y, 0.03, 0.02; weights = w, tol = 1e-10)
  @test sparsefit.converged && densefit.converged
  @test sparsefit.B == densefit.B
  @test sparsefit.weights == w
  @test sparse_nuclear_regression(ones(Int, 3, 2), ones(Int, 3, 1), 1, 1).B isa Matrix{Float64}
  short = sparse_nuclear_regression(X, Y, 0.001, 0.001; max_iter = 1, tol = 1e-14)
  @test !short.converged && short.iterations == 1
  for bad in (-1.0, Inf, NaN)
    @test_throws ArgumentError sparse_nuclear_regression(X, Y, bad, 0.1)
    @test_throws ArgumentError sparse_nuclear_regression(X, Y, 0.1, bad)
  end
  for bad in (0.0, -1.0, Inf, NaN)
    @test_throws ArgumentError sparse_nuclear_regression(X, Y, 0.1, 0.1; tol = bad)
    @test_throws ArgumentError sparse_nuclear_regression(X, Y, 0.1, 0.1; weights = fill(bad, 4))
  end
  @test_throws ArgumentError sparse_nuclear_regression(X, Y, 0.1, 0.1; max_iter = 0)
  @test_throws ArgumentError sparse_nuclear_regression(X, Y, 0.1, 0.1; sparsity = :bad)
  @test_throws DimensionMismatch sparse_nuclear_regression(X, Y, 0.1, 0.1; weights = ones(3))
  @test_throws DimensionMismatch sparse_nuclear_regression(X, Y[1:3, :], 0.1, 0.1)
  @test_throws ArgumentError sparse_nuclear_regression(zeros(0, 4), zeros(0, 3), 0.1, 0.1)
  @test_throws ArgumentError sparse_nuclear_regression(fill(NaN, 24, 4), Y, 0.1, 0.1)
  @test_throws ArgumentError sparse_nuclear_regression(X, fill(Inf, 24, 3), 0.1, 0.1)
  for bad in ([], [-1.0], [NaN], [Inf], [0.1, 0.2])
    @test_throws ArgumentError PenaltyGrid(sparse_nuclear_regression, X, Y; λ1 = bad, λ2 = [0.1])
    @test_throws ArgumentError PenaltyGrid(sparse_nuclear_regression, X, Y; λ1 = [0.1], λ2 = bad)
  end
  @test_throws ArgumentError PenaltyGrid(sparse_nuclear_regression, X, Y; nλ = (0, 2))
  @test_throws ArgumentError PenaltyGrid(sparse_nuclear_regression, X, Y; nλ = (1,))
  @test_throws ArgumentError PenaltyGrid(sparse_nuclear_regression, X, Y; λminratio = (0, 0.1))
  @test_throws ArgumentError CVPenaltyGrid(sparse_nuclear_regression, X, Y; folds = ones(Int, 24))
  @test_throws ArgumentError CVPenaltyGrid(sparse_nuclear_regression, X, Y; folds = [0; repeat(1:2, 11); 1])
  @test_throws DimensionMismatch CVPenaltyGrid(sparse_nuclear_regression, X, Y; folds = [1, 2])
  @test isinf(HDM._sn_relative_gap(1.0, 2.0))
  @test isinf(HDM._sn_relative_gap(Inf, 1.0))
  @test isinf(HDM._sn_relative_gap(1.0, NaN))
  @test HDM._sn_relative_gap(1.0, 1.0 + eps()) == 0
  @test HDM._sn_relative_gap(0.0, 0.0) == 0
end

end
end
