@testset "Optional PANOC loading" begin
  if Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt) === nothing
    @test_throws ArgumentError nuclear_regression(ones(3, 2), ones(3, 2), 0.1; solver = :panoc)
  end
end

using ProximalAlgorithms
import ProximalBase

@testset "Low-rank solver choices and diagnostics" begin
  rng = Xoshiro(81082)
  X = randn(rng, 60, 8)
  Y = X * randn(rng, 8, 2) * randn(rng, 2, 5) + 0.1randn(rng, 60, 5)
  @test Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt) !== nothing
  for solver in (:apg, :panoc)
    Q = Matrix(qr(randn(rng, 30, 6)).Q)[:, 1:6] * sqrt(30)
    Z = randn(rng, 30, 4)
    F = svd(Q'Z/30)
    for λ in (0.0, 0.05, 2first(F.S))
      s = nuclear_regression(Q, Z, λ; solver, intercept = false, tol = 1e-9)
      target = F.U * Diagonal(max.(F.S .- λ, 0)) * F.Vt
      @test s.converged
      @test s.B ≈ target atol = 1e-8
      @test s.objective ≈ sum(abs2, Z - Q*s.B)/60 + λ*sum(svdvals(s.B))
    end
  end
  for A in (X, hcat(X, X[:, 1:2]), randn(rng, 60, 80))
    apg = nuclear_regression(A, Y, 0.2; tol = 1e-9)
    panoc = nuclear_regression(A, Y, 0.2; solver = :panoc, tol = 1e-9)
    @test apg.converged && panoc.converged
    @test panoc.B ≈ apg.B atol = 2e-6 rtol = 2e-6
    @test predict(panoc, A) + residuals(panoc) ≈ Y
  end
  @test nuclear_regression(zeros(4, 3), ones(4, 2), 1; solver = :panoc).converged
  @test !nuclear_regression(X, Y, 0.1; solver = :panoc, max_iter = 1, tol = 1e-14).converged
  ns = nuclear_regression(Float32.(X), Float32.(Y), 0.2; solver = :panoc, tol = 1e-5)
  @test ns.converged && ns.B isa Matrix{Float32}

  λs = [1000.0, 0.2, 0.05]
  path = LowRankPath(X, Y; solver = :panoc, λpath = λs, tol = 1e-9)
  @test path.solutions[1].rank == 0
  for (λ, s) in zip(λs, path.solutions)
    @test s.converged
    @test s.B ≈ nuclear_regression(X, Y, λ; solver = :panoc, tol = 1e-9).B atol = 2e-6
  end

  Q = Matrix(qr(randn(rng, 30, 6)).Q)[:, 1:6] * sqrt(30)
  Z = randn(rng, 30, 4)
  C = Q'Z/30
  w = [1.0, 2.0, 0.5, 0.7, 3.0, 1.1]
  for λ in (0.0, 0.1, 100.0)
    target = copy(C)
    for j in axes(C, 1)
      target[j, :] .*= max(0, 1 - λ*w[j]/norm(C[j, :]))
    end
    s = sparse_reduced_rank(Q, Z, 4, λ; block_solver = :fista,
      weights = w, intercept = false, tol = 1e-10)
    @test s.converged
    @test s.B ≈ target atol = 1e-8
  end
  sp = LowRankPath(X, Y; method = :sparse, rank = 2, block_solver = :fista,
    λpath = λs, adaptive = true, tol = 1e-8)
  @test all(s -> s.converged, sp.solutions)
  @test sp.solutions[1].rank == 0
  @test sp.solutions[end].rank > 0
  @test sparse_reduced_rank(zeros(5, 3), ones(5, 2), 0, 1; block_solver = :fista).rank == 0
  fs = sparse_reduced_rank(Float32.(Q), Float32.(Z), 4, 0.1;
    block_solver = :fista, tol = 1e-5, inner_tol = 1e-5, intercept = false)
  @test fs.converged && fs.B isa Matrix{Float32}
  folds = repeat(1:3, 20)
  @test all(isfinite, CVLowRankPath(X, Y; solver = :panoc, λpath = [0.3, 0.1], folds).cvm)
  @test all(isfinite, CVLowRankPath(X, Y; method = :sparse, rank = 2,
    block_solver = :fista, λpath = [0.3, 0.1], folds, adaptive = true).cvm)
  @test !sparse_reduced_rank(X, Y, 2, 0.5; block_solver = :fista,
    max_iter = 1, inner_max_iter = 1, tol = 1e-14, inner_tol = 1e-14).converged

  for T in (Float32, Float64)
    for (A, Z) in ((X, Y), (hcat(X, X[:, 1:2]), Y), (randn(rng, 12, 20), randn(rng, 12, 4)),
                   (X, zeros(60, 5)), (Matrix{Float64}(I, 5, 5), Diagonal([2.0, 1.0, 0, 0, 0])))
      A, Z = T.(A), Matrix{T}(Z)
      path = LowRankPath(A, Z; method = :rank)
      for s in path.solutions
        @test s.rank == rank(s.B)
        k = isempty(s.activerows) ? 0 : rank((A .- mean(A; dims = 1))[:, s.activerows])
        @test dof(s) == s.rank*(k + size(Z, 2) - s.rank) + size(Z, 2)
        @test residuals(s) ≈ Z - predict(s, A) atol = 50eps(T)*max(one(T), norm(Z))
        @test s.objective ≈ sum(abs2, residuals(s))/(2size(A, 1))
      end
    end
  end
  A = Diagonal([1.0, 1e-8]) * [1.0 1.0; 1.0 -1.0]
  Z = [1.0 2.0 3.0; 0 0 0]
  s = reduced_rank(A, Z, 1; intercept = false, rtol = 0.1)
  @test s.rank == rank(s.B)
  @test dof(s) == s.rank*(rank(A[:, s.activerows]) + 3 - s.rank)
  selected = select_rank(X, Y; μ = 0.1)
  @test selected.solution.B ≈ reduced_rank(X, Y, selected.rank).B
  @test_throws ArgumentError nuclear_regression(X, Y, 1; solver = :unknown)
  @test_throws ArgumentError sparse_reduced_rank(X, Y, 2, 1; block_solver = :unknown)
  @test_throws ArgumentError LowRankPath(X, Y; solver = :unknown)
  @test_throws ArgumentError LowRankPath(X, Y; method = :rank, solver = :panoc)
  @test_throws ArgumentError LowRankPath(X, Y; method = :nuclear, block_solver = :fista)
end

@testset "Low-rank solver boundary behavior and StatsAPI entry points" begin
  # A single sweep on a coupled quadratic is not a converged block solve.
  H = [2.0 1.0; 1.0 2.0]
  C = ones(2, 1)
  A = zeros(2, 1)
  @test !HighDimensionalModels._lr_group!(A, H, C, 0.0, ones(2);
    max_iter = 1, tol = 1e-12)
  @test vec(A) ≈ [0.5, 0.25]
  @test norm(H*A - C) > 1e-12

  fill!(A, 0)
  Cf = reshape([1.0, 0.0], 2, 1)
  @test !HighDimensionalModels._lr_group_fista!(A, H, Cf, 0.0, ones(2);
    max_iter = 1, tol = 1e-12, L = 3.0)
  @test norm(H*A - Cf) > 1e-12

  # A zero quadratic has the zero minimum-norm solution, including with a
  # nonzero warm start. The final KKT check still rejects an inconsistent C.
  fill!(A, 2)
  @test HighDimensionalModels._lr_group_fista!(A, zeros(2, 2), zeros(2, 1),
    0.1, ones(2); max_iter = 1, tol = 1e-12, L = 0.0)
  @test iszero(A)
  @test !HighDimensionalModels._lr_group_fista!(A, zeros(2, 2), C,
    0.1, ones(2); max_iter = 1, tol = 1e-12, L = 0.0)

  u = zeros(2)
  @test !HighDimensionalModels._lr_lasso_gram!(u, H, ones(2), 0.0;
    max_iter = 1, tol = 1e-12)
  @test u ≈ [0.5, 0.25]

  # Both ways to remove a sparse component must return a valid zero fit.
  d = HighDimensionalModels._lr_data(Matrix{Float64}(I, 2, 2), zeros(2, 2), false)
  B, ok, _, df = HighDimensionalModels._lr_unit(d, 0.1, ones(2), ones(2),
    ones(2), [1.0, 0.0]; max_iter = 2, tol = 1e-12,
    inner_max_iter = 1, inner_tol = 1e-12)
  @test ok && iszero(B) && df == 0

  # Correlated predictors can leave nonzero u after one inexact sweep even
  # for Y=0. The exact response update must then remove the whole component.
  d = HighDimensionalModels._lr_data(ones(2, 2), zeros(2, 2), false)
  B, ok, _, df = HighDimensionalModels._lr_unit(d, 0.1, ones(2), ones(2),
    ones(2), [1.0, 0.0]; max_iter = 2, tol = 1e-12,
    inner_max_iter = 1, inner_tol = 1e-12)
  @test ok && iszero(B) && df == 0

  rng = Xoshiro(81083)
  X, Y = randn(rng, 24, 3), randn(rng, 24, 2)
  @test_throws ArgumentError LowRankPath(X, Y; weights = ones(3))
  @test_throws ArgumentError LowRankPath(X, Y; adaptive = true)
  p = fit(LowRankPath, X, Y; method = :rank, ranks = 0:2)
  @test p isa LowRankPath
  @test p.ranks == [0, 1, 2]
  @test p.solutions[end].B ≈ reduced_rank(X, Y, 2).B
  cv = fit(CVLowRankPath, X, Y; method = :rank, ranks = 0:2,
    folds = repeat(1:3, 8))
  @test cv isa CVLowRankPath
  @test predict(cv, X) ≈ predict(cv.path.solutions[cv.index], X)

  # The optional ProximalAlgorithms bridge must expose the penalty value as
  # well as its prox: lambda * sum of the coefficient singular values.
  ext = Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt)
  penalty = ext.NuclearPenalty(ProximalBase.NuclearNorm(0.2))
  @test penalty([3.0 0.0; 0.0 1.0]) ≈ 0.8
end


@testset "Exact rank degrees of freedom preserve response scale" begin
  # For X=I and singular values (2,1), rank-one Stein df is
  # 1*(2+2-1) + 2/(4-1) = 11/3, independent of response units.
  for T in (Float32, Float64)
    X = Matrix{T}(I, 2, 2)
    scales = T === Float32 ? T[1e-12, 1, 1e12] : T[1e-100, 1, 1e100]
    for scale in scales
      Y = Matrix(Diagonal(T[2, 1] .* scale))
      s = reduced_rank(X, Y, 1; intercept = false)
      @test isfinite(s.df_exact)
      @test s.df_exact ≈ T(11)/T(3)
      @test isfinite(lowrank_ic(s; df = :exact))
    end
    tied = reduced_rank(X, X, 1; intercept = false)
    @test isinf(tied.df_exact)
    @test isinf(lowrank_ic(tied; df = :exact))
  end
end

include("lowrank_panoc_costs.jl")
