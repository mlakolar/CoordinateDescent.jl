module MultiLassoTest

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics, Test

function kkt_rows(G, B, λ, ω)
  maximum(axes(B, 1)) do k
    nb = norm(B[k, :])
    iszero(nb) ? max(norm(G[k, :]) - λ * ω[k], 0.0) : norm(G[k, :] + λ * ω[k] * B[k, :] / nb)
  end
end

@testset "multi_lasso and MultiLassoPath" begin
  rng = Xoshiro(41)
  n, p, q = 200, 30, 3
  X = randn(rng, n, p) .* (1 .+ 2 .* rand(rng, p))'
  B = zeros(p, q); B[[1, 4, 7], :] .= randn(rng, 3, q)
  Y = 1.0 .+ X * B .+ randn(rng, n, q)
  ω = vec(std(X; dims = 1, corrected = false))

  sol = multi_lasso(X, Y, 0.1)
  @test sol.converged && size(sol.B) == (p, q) && length(sol.intercept) == q
  R = Y .- sol.intercept' .- X * Matrix(sol.B)
  @test norm(mean(R; dims = 1)) < 1e-12
  @test kkt_rows(-(X .- mean(X; dims = 1))' * R / n, Matrix(sol.B), 0.1, ω) < 1e-6
  @test sol.activerows == [k for k in 1:p if norm(sol.B[k, :]) > 0]
  @test issubset([1, 4, 7], sol.activerows)
  # one response: the lasso
  s1 = multi_lasso(X, Y[:, 1:1], 0.1)
  l1 = lasso(X, Y[:, 1], 0.1; intercept = true, standardize_x = true)
  @test vec(Matrix(s1.B)) ≈ Vector(l1.x) atol = 1e-6
  @test s1.intercept[1] ≈ l1.intercept atol = 1e-6
  # no standardization, no intercept
  s0 = multi_lasso(X, Y .- 1, 0.1; intercept = false, standardize_x = false)
  @test s0.intercept == zeros(q)
  @test kkt_rows(-X' * (Y .- 1 .- X * Matrix(s0.B)) / n, Matrix(s0.B), 0.1, ones(p)) < 1e-6
  # warm_start = false: along num_steps levels from λmax
  sw = multi_lasso(X, Y, 0.1, CDOptions(; warm_start = false, num_steps = 15))
  @test Matrix(sw.B) ≈ Matrix(sol.B) atol = 1e-5
  # integer data
  @test multi_lasso(round.(Int, X), round.(Int, Y), 1).converged
  @test_throws ArgumentError multi_lasso(X, Y, -0.1)
  @test_throws DimensionMismatch multi_lasso(X, Y[1:end-1, :], 0.1)

  # paths
  path = MultiLassoPath(X, Y; nλ = 25, λminratio = 0.01)
  @test length(path.λpath) == 25 && all(path.converged) && path.intercept
  @test nnz(path.Bpath[1]) == 0 && !isempty(path.activerows[2])
  for k in (6, 25)
    s = multi_lasso(X, Y, path.λpath[k], CDOptions(; opt_tol = 1e-10, gaptol = 0))
    @test Matrix(path.Bpath[k]) ≈ Matrix(s.B) atol = 1e-5
    @test path.β0path[k] ≈ s.intercept atol = 1e-5
  end
  path0 = MultiLassoPath(X, Y, path.λpath; screening = false)
  @test all(norm(path.Bpath[k] - path0.Bpath[k]) < 1e-5 for k in eachindex(path.λpath))
  pathm = MultiLassoPath(X, Y; nλ = 50, max_rows = 4)
  @test length(last(pathm.activerows)) > 4 && all(length(a) <= 4 for a in pathm.activerows[1:end-1])
  # integer data, and a range of levels
  @test MultiLassoPath(round.(Int, X), round.(Int, Y); nλ = 5).λpath isa Vector{Float64}
  pr = MultiLassoPath(X, Y, range(path.λpath[1], path.λpath[10]; length = 4))
  @test length(pr.λpath) == 4 && pr.λpath[4] ≈ path.λpath[10]
end

end
