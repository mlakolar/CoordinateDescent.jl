module SelectionTests
using HighDimensionalModels, ProximalBase, ProximalSolvers, LinearAlgebra, Random, SparseArrays, Statistics, Test

@testset "SLOPE orthogonal solution and scale" begin
  rng = Xoshiro(82)
  n, p = 80, 12
  X = sqrt(n) * Matrix(qr(randn(rng, n, p)).Q)
  y = X * [3., -2., 1., zeros(p-3)...] + randn(rng, n)
  λ = slope_bh(p; q=0.1, sigma=1, n)
  sol = slope(X, y, λ; tol=1e-9)
  expected = zeros(p)
  prox!(expected, ProximalBase.SLOPE(λ), X'y/n, 1.)
  @test sol.converged
  @test sol.x ≈ expected atol=1e-7
  @test residuals(sol) ≈ y - predict(sol, X)
  @test deviance(sol) ≈ sum(abs2, residuals(sol))
  @test nobs(sol) == n
  @test coef(sol) == sol.x
  @test slope(X, y; q=0.1, tol=1e-9).x ≈ expected atol=1e-7
  @test slope(X, y, fill(0.2, p); tol=1e-9).x ≈ Vector(lasso(X,y,0.2; opt_tol=1e-10).x) atol=1e-6
  @test slope_bh(2; q=0.1)[1] ≈ 1.959963984540054
  # Orthogonal after centering; nontrivial input units and intercept.
  Z = X .+ 3
  Z[:, 1] .*= 5
  a = slope(Z, y .+ 4, λ; intercept=true, standardize_x=true, tol=1e-8)
  Zc = Z .- mean(Z; dims=1)
  scales = vec(sqrt.(sum(abs2, Zc; dims=1)/n))
  b = slope(Zc ./ transpose(scales), y .- mean(y), λ; tol=1e-8)
  @test a.converged && b.converged
  @test a.x .* scales ≈ b.x atol=1e-6
  @test mean(residuals(a)) ≈ 0 atol=1e-10
  @test coef(a)[1] == a.intercept
  @test slope(sparse(X),y,λ).x ≈ sol.x atol=1e-5
  @test slope(Float32.(X),Float32.(y),Float32.(λ); tol=1e-4).converged
  @test_throws ArgumentError slope_bh(0)
  @test_throws ArgumentError slope_bh(p; sigma=NaN)
  @test_throws ArgumentError slope(X,y,reverse(λ))
  @test_throws ArgumentError slope(X,y,fill(NaN,p))
  @test_throws DimensionMismatch slope(X,y,λ[2:end])
  @test_throws ArgumentError slope(X,y,λ; max_iter=1)
  @test_throws DimensionMismatch predict(sol,zeros(2,p+1))
end

@testset "SLOPE screening and KKT repair" begin
  for seed in 1:4
    rng = Xoshiro(seed)
    X = randn(rng,60,90)
    X[:,2:10] .+= 0.8 .* X[:,1]
    y = 2X[:,1] - X[:,20] + randn(rng,60)
    a = SLOPEPath(X,y; nλ=12,λmin_ratio=0.15,intercept=true,tol=2e-7)
    b = SLOPEPath(X,y,a.λpath;lambda=a.lambda,screening=false,intercept=true,tol=2e-7)
    @test all(a.converged) && all(b.converged)
    @test predict(a,X) ≈ predict(b,X) atol=2e-4
    @test norm(a.βpath[1],Inf) < 1e-6
    @test nobs(a) == 60
    @test size(coef(a)) == (91,12)
  end
  X = Matrix{Float64}(I,4,4) * 2
  y = [3.,2.,1.,0.]
  lambda = [0.4,0.3,0.2,0.1]
  factors = [1.,.5,.2]
  wrapped = SLOPEPath(X,y,factors;lambda,tol=1e-9)
  shared = ProximalSolvers.solve_path((1/4)*LeastSquaresLoss(y,X),
    ProximalBase.SLOPE(lambda),factors,ProximalSolvers.SLOPEAPG(;tol=1e-9))
  @test wrapped.βpath == shared.solutions
  @test wrapped.iterations == shared.iterations
  @test wrapped.converged == shared.converged
  @test wrapped.kkt_residuals == shared.kkt_residuals
  @test wrapped.working_sizes == shared.working_sizes
  @test wrapped.kkt_repairs == shared.kkt_repairs
  @test_throws ArgumentError SLOPEPath(X,y,[0.1,1.])
  @test_throws ArgumentError SLOPEPath(X,y;lambda=zeros(4))
  z = SLOPEPath(X,zeros(4);nλ=1)
  @test all(z.converged) && all(iszero,coef(z))
end

@testset "Knockoff construction, threshold and statistic" begin
  rng = Xoshiro(32)
  p, n = 8, 90
  S = [0.4^abs(i-j) for i in 1:p,j in 1:p]
  S .*= (collect(1.:p) * transpose(collect(1.:p)))
  s = equicorrelated_s(S)
  @test minimum(eigvals(Symmetric(2S-Diagonal(s)))) > -1e-10
  X = randn(rng,n,p) * cholesky(S).U
  Xk = fixed_knockoffs(X)
  ds = equicorrelated_s(X'X)
  @test Xk'Xk ≈ X'X atol=1e-8
  @test X'Xk ≈ X'X-Diagonal(ds) atol=1e-8
  K = gaussian_knockoffs(X,zeros(p),S;noise=zeros(n,p))
  @test K ≈ X-X*(S\Diagonal(s))
  @test gaussian_knockoffs(X,zeros(p),S;rng=Xoshiro(1)) == gaussian_knockoffs(X,zeros(p),S;rng=Xoshiro(1))
  @test knockoff_threshold([0.,0.]) == Inf
  @test knockoff_threshold([4.,3.,2.,-1.];q=0.4) == 2
  @test knockoff_threshold([2.,2.,-2.];q=0.7) == Inf
  @test knockoff_threshold([2.,2.,-2.];q=0.7,offset=0) == 2
  @test knockoff_threshold(Float64[]) == Inf
  y = X[:,1] .+ randn(rng,n)
  W = lasso_coefdiff(X,Xk,y,0.1; opt_tol=1e-10,order=:cyclic)
  a,b = copy(X),copy(Xk)
  a[:,3],b[:,3] = Xk[:,3],X[:,3]
  Wswap = lasso_coefdiff(a,b,y,0.1;opt_tol=1e-10,order=:cyclic)
  @test Wswap[3] ≈ -W[3] atol=1e-7
  @test Wswap[[1,2,4,5,6,7,8]] ≈ W[[1,2,4,5,6,7,8]] atol=1e-7
  result = knockoff_filter(X,y,0.1;method=:provided,Xk,q=0.5)
  @test result.selected == findall(>=(result.threshold), result.W)
  # Public construction routes must agree with the same supplied knockoffs.
  # Cyclic coordinate updates keep the statistic independent of RNG consumption
  # during Gaussian construction.
  for method in (:fixed, :gaussian)
    built = knockoff_filter(X,y,0.1;method,mu=zeros(p),Sigma=S,q=0.5,
      rng=Xoshiro(91),order=:cyclic,opt_tol=1e-10)
    K = method === :fixed ? fixed_knockoffs(X) :
      gaussian_knockoffs(X,zeros(p),S;rng=Xoshiro(91))
    supplied = knockoff_filter(X,y,0.1;method=:provided,Xk=K,q=0.5,
      order=:cyclic,opt_tol=1e-10)
    @test built.Xk ≈ K
    @test built.W ≈ supplied.W atol=1e-9
    @test built.selected == supplied.selected
    @test built.method === method
  end
  @test_throws ArgumentError fixed_knockoffs(randn(10,8))
  @test_throws ArgumentError gaussian_knockoffs(X,zeros(p),ones(p,p))
  @test_throws ArgumentError gaussian_knockoffs(X,zeros(p),S;s=fill(1e8,p))
  @test_throws ArgumentError knockoff_threshold([NaN])
  @test_throws ArgumentError knockoff_filter(X,y,0.1)
  @test_throws ArgumentError knockoff_filter(X,y,0.1;method=:fixed,Xk)
end

@testset "Knockoff feasibility under changes of units" begin
  for T in (Float32,Float64), scale in (T(1e-8),one(T),T(1e8))
    X = reshape(scale .* T[1,2],:,1)
    S = X'X
    bad = 3 .* diag(S)
    @test_throws ArgumentError fixed_knockoffs(X;s=bad)
    @test_throws ArgumentError gaussian_knockoffs(X,T[0],S;s=bad)
    for ratio in (zero(T),one(T),T(2))
      s = ratio .* diag(S)
      K = fixed_knockoffs(X;s)
      @test (K'K)[1]/S[1] ≈ one(T) atol=100eps(T)
      @test (X'K)[1]/S[1] ≈ one(T)-ratio atol=100eps(T)
      _,A,C = HighDimensionalModels._knockoff_parameters(S,s)
      @test (C'C)[1]/S[1] ≈ 2ratio-ratio^2 atol=100eps(T)
      @test gaussian_knockoffs(X,T[0],S;s,noise=zeros(T,2,1)) ≈ X-X*A
    end
  end
  # A large variance in another column must not mask an infeasible small one.
  X = [1e-8 0.; 0. 1.; 0. 0.; 0. 0.]
  S = X'X
  @test_throws ArgumentError fixed_knockoffs(X;s=[3e-16,0.])
  @test_throws ArgumentError gaussian_knockoffs(X,zeros(2),S;s=[3e-16,0.])
end

@testset "Complementary pairs and deterministic parallel fitting" begin
  rng = Xoshiro(83)
  X = randn(rng,61,20)
  y = X[:,1] + X[:,2] + randn(rng,61)
  pairs = complementary_pairs(61,7;rng=Xoshiro(1))
  @test all(pair -> length(pair[1])==length(pair[2])==30 && isempty(intersect(pair...)),pairs)
  @test pairs == complementary_pairs(61,7;rng=Xoshiro(1))
  a = stability_selection(X,y,0.15;q=4,pairs,parallel=false,rng=Xoshiro(2))
  b = stability_selection(X,y,0.15;q=4,pairs,parallel=true,rng=Xoshiro(2))
  @test a.selections == b.selections
  @test all(sum(a.selections;dims=1) .<= 4)
  @test a.frequencies == vec(sum(a.selections;dims=2))/14
  @test a.selected == findall(>=(0.9),a.frequencies)
  @test all(a.pair_frequencies .>= 2 .* a.frequencies .- 1 .- eps())
  @test a.pfer_bound ≈ 16/(20*0.8)
  @test stability_bound(100,5,0.9;theta=0.1) ≈ 0.625
  @test stability_selection(X,y,0.15;q=0,npairs=2).selected == Int[]
  @test_throws ArgumentError stability_selection(X,y,0.15;q=21)
  @test_throws ArgumentError stability_selection(X,y,0.15;q=4,pairs=[([1,2],[2,3])])
  @test_throws ArgumentError stability_bound(20,4,0.5)
  @test_throws ArgumentError complementary_pairs(1,4)
end
end
