@testset "Optional PANOC loading" begin
  if Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt) === nothing
    @test_throws ArgumentError nuclear_regression(ones(3, 2), ones(3, 2), 0.1; solver = :panoc)
  end
end

using ProximalAlgorithms

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
