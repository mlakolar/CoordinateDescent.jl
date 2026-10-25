module GroupLassoTest

using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# the largest violation of the optimality conditions of λ ∑ₖ wₖ ‖θ_Gₖ‖ for the
# gradient G
function kkt_group(G, θ, groups, λ, w)
  maximum(enumerate(groups)) do (k, g)
    nθ = norm(θ[g])
    iszero(nθ) ? max(norm(G[g]) - λ * w[k], 0.0) : norm(G[g] + λ * w[k] * θ[g] / nθ)
  end
end

@testset "group_lasso and GroupLassoPath" begin
  rng = Xoshiro(21)
  n, J = 200, 12
  sizes = [3, 2, 4, 1, 3, 2, 5, 3, 2, 4, 1, 3]
  p = sum(sizes)
  ends = cumsum(sizes)
  groups = [ends[k]-sizes[k]+1:ends[k] for k in 1:J]
  X = randn(rng, n, p)
  X[:, 2] .= X[:, 1] .+ 0.5 .* X[:, 2]                         # correlated columns within a group
  X .*= (1 .+ 3 .* rand(rng, p))'                               # on different scales
  β = zeros(p); β[groups[1]] .= [1.0, -0.5, 0.5]; β[groups[3]] .= 0.3; β[groups[7][1:2]] .= [-0.6, 0.4]
  y = 2.0 .+ X * β .+ randn(rng, n)
  w = sqrt.(sizes)
  λ = 0.1

  @testset "orthonormal groups" begin
    sol = group_lasso(X, y, λ, groups)
    @test sol.converged
    @test sol.groups == collect.(groups) && sol.groupweights ≈ w
    # the problem in θ_G = R̃_G β_G on the orthonormalized design Z
    xbar = mean(X; dims = 1)
    Zs, Rs = Matrix{Float64}[], Matrix{Float64}[]
    for G in groups
      F = qr(X[:, G] .- xbar[:, G])
      push!(Zs, sqrt(n) .* Matrix(F.Q)); push!(Rs, Matrix(F.R) ./ sqrt(n))
    end
    Z = reduce(hcat, Zs)
    θ = reduce(vcat, [Rs[k] * Vector(sol.x[G]) for (k, G) in enumerate(groups)])
    r = y .- sol.intercept .- X * Vector(sol.x)
    @test abs(mean(r)) < 1e-10
    @test kkt_group(-Z' * r / n, θ, groups, λ, w) < 1e-6
    # the same solution as coordinate descent on Z, mapped back
    xz = AtomIterate(p, groups)
    coordinate_descent!(xz, CDLeastSquaresLoss(y, Z; intercept = true), GroupNormL2(λ, groups, w), CDOptions(; opt_tol = 1e-10, gaptol = 0))
    βz = reduce(vcat, [Rs[k] \ xz[G] for (k, G) in enumerate(groups)])
    @test Vector(sol.x) ≈ βz atol = 1e-5
    @test sol.activegroups == [k for (k, G) in enumerate(groups) if norm(βz[G]) > 0]
    # invariant to a change of basis within a group: scaling a column scales
    # its coefficient and nothing else
    Xs = copy(X); Xs[:, 5] .*= 7
    sols = group_lasso(Xs, y, λ, groups)
    @test sols.x[5] ≈ sol.x[5] / 7 rtol = 1e-5
    @test Vector(sols.x)[[1:4; 6:p]] ≈ Vector(sol.x)[[1:4; 6:p]] atol = 1e-6
    @test sols.intercept ≈ sol.intercept atol = 1e-6
    # a column that is a combination of the others of its group gets zero, and
    # the fit is the one without it
    Xd = [X X[:, 1] .- 2 .* X[:, 3]]
    gd = [i == 1 ? [1, 2, 3, p + 1] : collect(G) for (i, G) in enumerate(groups)]
    sold = group_lasso(Xd, y, λ, gd; groupweights = w)            # the weights of the groups without it
    @test Xd * Vector(sold.x) ≈ X * Vector(sol.x) atol = 1e-6
    @test count(!iszero, Vector(sold.x)[[1, 2, 3, p + 1]]) <= 3
    # a zero group weight keeps a group in the model
    wz = copy(w); wz[12] = 0
    solz = group_lasso(X, y, 0.5, groups; groupweights = wz)
    @test 12 in solz.activegroups
    # without an intercept
    sol0 = group_lasso(X, y .- 2, λ, groups; intercept = false)
    @test sol0.intercept == 0
    # warm_start = false: along num_steps levels from λmax
    solw = group_lasso(X, y, λ, groups, CDOptions(; warm_start = false, num_steps = 20))
    @test Vector(solw.x) ≈ Vector(sol.x) atol = 1e-5
    @test_throws ArgumentError group_lasso(X, y, λ, groups[1:end-1])          # a column in no group
    @test_throws ArgumentError group_lasso(X, y, λ, [groups; [1:1]])          # a column in two groups
    @test_throws ArgumentError group_lasso(X, y, -1.0, groups)
    @test_throws ArgumentError group_lasso(X, y, λ, groups; standardize = :bogus)
  end

  @testset "columns as given" begin
    sol = group_lasso(X, y, λ, groups; standardize = :none)
    r = y .- sol.intercept .- X * Vector(sol.x)
    Xc = X .- mean(X; dims = 1)
    @test kkt_group(-Xc' * r / n, Vector(sol.x), groups, λ, w) < 1e-6
    # groups need not be contiguous
    perm = randperm(rng, p)
    gp = [perm[G] for G in groups]
    solp = group_lasso(X[:, invperm(perm)], y, λ, gp; standardize = :none)
    @test Vector(solp.x)[perm] ≈ Vector(sol.x) atol = 1e-5
    # integer data
    soli = group_lasso(round.(Int, 10X), round.(Int, 10y), 1.0, groups)
    @test soli.converged
  end

  @testset "paths" begin
    # gaptol = 0: each level stops on the changes, for the near-exact
    # comparisons with the direct solves below
    exact = CDOptions(; gaptol = 0)
    path = GroupLassoPath(X, y, groups, exact; nλ = 30, λminratio = 0.01)
    @test length(path.λpath) == 30 && all(path.converged) && path.intercept
    @test nnz(path.βpath[1]) == 0 && isempty(path.activegroups[1])
    @test !isempty(path.activegroups[2])                      # λmax is the smallest level with a zero solution
    @test all(path.kkt_checks .>= 1)
    for k in (8, 30)
      sol = group_lasso(X, y, path.λpath[k], groups, CDOptions(; opt_tol = 1e-10, gaptol = 0))
      @test Vector(path.βpath[k]) ≈ Vector(sol.x) atol = 1e-5
      @test path.β0path[k] ≈ sol.intercept atol = 1e-5
      @test path.activegroups[k] == sol.activegroups
    end
    path0 = GroupLassoPath(X, y, path.λpath, groups, exact; screening = false)
    @test all(norm(path.βpath[k] - path0.βpath[k]) < 1e-5 for k in eachindex(path.λpath))
    pathm = GroupLassoPath(X, y, groups; nλ = 50, max_groups = 3)
    @test length(last(pathm.activegroups)) > 3 && all(length(a) <= 3 for a in pathm.activegroups[1:end-1])
    pathn = GroupLassoPath(X, y, range(1.0, 0.1; length = 5), groups; standardize = :none)
    @test length(pathn.λpath) == 5
    # integer data
    @test GroupLassoPath(round.(Int, 10X), round.(Int, 10y), groups; nλ = 5).λpath isa Vector{Float64}
  end
end

end
