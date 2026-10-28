module KnockoffConstructionTests
using HighDimensionalModels, LinearAlgebra, Random, Statistics, Test
const H = HighDimensionalModels

@testset "construction selector and compatibility" begin
  rng=Xoshiro(106);X=randn(rng,30,4);y=randn(rng,30);Z=randn(rng,30,4)
  S=[.4^abs(i-j) for i in 1:4,j in 1:4];mu=zeros(4)
  X0,y0,Z0,S0=copy(X),copy(y),copy(Z),copy(S)
  for T in (Float32,Float64)
    St=T.(S);Xt=T.(X);mut=T.(mu);Zt=T.(Z)
    d,A,R=H._knockoff_parameters(St,nothing)
    expected=Xt-(Xt.-mut')*A+Zt*R
    plan=knockoff_diagonal(St)
    @test plan.strategy==:equicorrelated
    @test plan.s==d && eltype(plan.s)==eltype(d)
    @test gaussian_knockoffs(Xt,mut,St;noise=Zt)==expected
    @test gaussian_knockoffs(Xt,mut,St;noise=Zt,strategy=:equicorrelated)==expected
    @test plan.diagnostics.feasibility==:validated
    supplied=T.(0.5 .* diag(St));p=knockoff_diagonal(St;s=supplied)
    @test p.strategy==:user && p.s==supplied && p.s!==supplied
    supplied[1]=0
    @test p.s[1]>0
    K=fixed_knockoffs(Xt;strategy=:equicorrelated)
    d2,A2,R2=H._knockoff_parameters(Xt'Xt,nothing)
    E=zeros(T,30,4);for j in 1:4;E[4+j,j]=1;end
    @test K==Xt-Xt*A2+(qr(Xt).Q*E)*R2
    @test K'K≈Xt'Xt rtol=200eps(T)
    @test Xt'K≈Xt'Xt-Diagonal(d2) rtol=200eps(T)
  end
  @test X==X0 && y==y0 && Z==Z0 && S==S0
  @test gaussian_knockoffs(X,mu,S;rng=Xoshiro(17))==gaussian_knockoffs(X,mu,S;rng=Xoshiro(17))
  for strategy in (:bad,:user)
    @test_throws ArgumentError knockoff_diagonal(S;strategy)
  end
  for strategy in (:equicorrelated,:mvr,:maxent)
    @test_throws ArgumentError knockoff_diagonal(S;strategy,s=ones(4))
  end
  @test_throws ArgumentError knockoff_diagonal(S;construction_options=(unknown=1,))
  @test_throws ArgumentError knockoff_diagonal(S;strategy=:user,s=ones(4),construction_options=(tol=1e-8,))
  @test_throws DimensionMismatch knockoff_diagonal(S;s=ones(3))
  @test_throws ArgumentError knockoff_diagonal(S;s=-ones(4))
  @test_throws ArgumentError knockoff_diagonal(S;s=fill(3.,4))
  @test_throws ArgumentError knockoff_diagonal(S;s=fill(NaN,4))
  @test_throws ArgumentError knockoff_diagonal(S;construction_options=Dict())
  for strategy in (:mvr,:maxent),opts in ((niter=0,),(tol=0.,),(λmin=-1.,),(robust=1,),
      (optimality_tol=0.,),(optimality_tol=1.,),(optimality_tol=2.,),
      (optimality_tol=Inf,),(optimality_tol=NaN,),(unknown=1,))
    @test_throws ArgumentError knockoff_diagonal(S;strategy,construction_options=opts)
  end
  for strategy in (:mvr,:maxent),optimality_tol in (0.,1.,2.,Inf,NaN)
    @test_throws r"optimality_tol must be finite and in \(0, 1\)" knockoff_diagonal(S;strategy,construction_options=(;optimality_tol))
  end
  if Base.get_extension(H,:HighDimensionalModelsKnockoffsExt)===nothing
    @test_throws ArgumentError knockoff_diagonal(S;strategy=:mvr)
    @test_throws ArgumentError knockoff_diagonal(S;strategy=:maxent)
  end
  for scale in (1e-20,1.,1e20),T in (Float32,Float64)
    D=Matrix(Diagonal(T.(scale.*[.2,1,3,10])))
    for s in (zeros(T,4),2diag(D))
      p=knockoff_diagonal(D;s)
      @test p.s==s
      d,A,R=H._knockoff_parameters(D,s)
      @test R'R≈2Diagonal(d)-Diagonal(d)*A atol=100eps(T)*scale
      @test eigmin(Symmetric(2D-Diagonal(d)))>=-100eps(T)*scale
    end
  end
end

@testset "filter forwarding and conflict rejection" begin
  rng=Xoshiro(106);X=randn(rng,40,4);y=randn(rng,40);Z=randn(rng,40,4);mu=zeros(4)
  S=[.3^abs(i-j) for i in 1:4,j in 1:4];s=.4diag(S)
  before=map(copy,(X,y,mu,S,s,Z))
  K=gaussian_knockoffs(X,mu,S;s,noise=Z)
  f=knockoff_filter(X,y,.2;Sigma=S,mu,s,noise=Z,order=:cyclic,opt_tol=1e-9)
  @test f.Xk==K && f.method==:gaussian && f.construction.strategy==:user
  @test f.construction.s==s && f.construction.s!==s
  @test f.W==lasso_coefdiff(X,K,y,.2;order=:cyclic,opt_tol=1e-9)
  provided=knockoff_filter(X,y,.2;method=:provided,Xk=K,order=:cyclic,opt_tol=1e-9)
  @test provided.W==f.W && provided.construction.strategy==:provided
  old=KnockoffResult(f.selected,f.W,f.threshold,f.Xk,f.q,f.lambda,f.method)
  @test old.construction.strategy==:unspecified
  @test old.W===f.W && old.Xk===f.Xk
  typed=KnockoffResult{eltype(f.W),typeof(f.Xk)}(f.selected,f.W,f.threshold,f.Xk,f.q,f.lambda,f.method)
  @test typed.construction.strategy==:unspecified
  # Preserve legacy unused Gaussian parameters for other model routes.
  legacy=knockoff_filter(X,y,.2;method=:provided,Xk=K,Sigma=S,mu,order=:cyclic,opt_tol=1e-9)
  @test legacy.W==provided.W
  for kw in ((strategy=:equicorrelated,), (s=s,), (noise=Z,), (construction_options=(safety=.01,),))
    @test_throws ArgumentError knockoff_filter(X,y,.2;method=:provided,Xk=K,kw...)
  end
  @test_throws ArgumentError knockoff_filter(X,y,.2;method=:fixed,noise=Z)
  @test_throws ArgumentError knockoff_filter(X,y,.2;Sigma=S,mu,strategy=:mvr,s=s)
  @test_throws ArgumentError knockoff_filter(X,y,-.2;Sigma=S,mu)
  # Invalid solver fields fail before diagonal optimization or consuming RNG.
  rng=Xoshiro(17);control=copy(rng)
  @test_throws ArgumentError knockoff_filter(X,y,.2;Sigma=S,mu,rng,not_a_cd_option=true)
  @test rand(rng)==rand(control)
  @test all(isequal.(before,(X,y,mu,S,s,Z)))
end

@testset "fixed-X numerical rank before construction" begin
  for T in (Float32,Float64)
    duplicated=ones(T,6,2)
    zero_column=hcat(ones(T,6),zeros(T,6))
    near_duplicate=copy(duplicated);near_duplicate[4,2]+=eps(T)
    # The Float32 perturbation is the rounded value of 1 + 1f-7.
    for X in (duplicated,zero_column,near_duplicate)
      y=ones(T,6)
      @test_throws ArgumentError fixed_knockoffs(X)
      @test_throws ArgumentError knockoff_filter(X,y,T(.2);method=:fixed)
    end
    X=T[1 0;0 1;1 1;1 -1;2 1;-1 2]
    d,A,C=H._knockoff_parameters(X'X,nothing)
    E=zeros(T,6,2);E[3,1]=1;E[4,2]=1
    expected=X-X*A+(qr(X).Q*E)*C
    @test fixed_knockoffs(X)==expected
    result=knockoff_filter(X,ones(T,6),T(.2);method=:fixed,order=:cyclic)
    @test result.Xk==expected
  end
end

@testset "Gaussian joint covariance with nonzero mean" begin
  rng=Xoshiro(10603);n=20000
  Sigma=[1.0 .4 .1;.4 1.5 .3;.1 .3 .8];mu=[1.5,-2.,.75]
  X=randn(rng,n,3)*cholesky(Symmetric(Sigma)).U .+ mu'
  Z=randn(rng,n,3)
  for strategy in (:equicorrelated,:user)
    supplied=strategy===:user ? (;s=.35diag(Sigma)) : (;)
    plan=knockoff_diagonal(Sigma;strategy,supplied...)
    K=gaussian_knockoffs(X,mu,Sigma;strategy,noise=Z,supplied...)
    D=Diagonal(plan.s);target=[Sigma Sigma-D;Sigma-D Sigma]
    @test all(isapprox.(vec(mean(hcat(X,K);dims=1)),vcat(mu,mu);atol=.04,rtol=0))
    @test all(isapprox.(cov(hcat(X,K)),target;atol=.06,rtol=0))
  end
end

@testset "filter sampling and solver share the RNG" begin
  data_rng=Xoshiro(10604);X=randn(data_rng,80,4);y=X[:,1]+randn(data_rng,80)
  mu=[.2,-.1,.3,-.4];Sigma=[.3^abs(i-j) for i in 1:4,j in 1:4];s=.4diag(Sigma)
  before=map(copy,(X,y,mu,Sigma,s))
  for strategy in (:equicorrelated,:user)
    supplied=strategy===:user ? (;s) : (;)
    sampled_rng=Xoshiro(17);noise_rng=Xoshiro(17)
    # Construction draws first; pass the advanced generator to the random-order solver.
    noise=randn(noise_rng,Float64,size(X)...)
    sampled=knockoff_filter(X,y,.2;mu,Sigma,strategy,rng=sampled_rng,order=:random,supplied...)
    supplied_noise=knockoff_filter(X,y,.2;mu,Sigma,strategy,noise,rng=noise_rng,order=:random,supplied...)
    @test sampled.Xk==supplied_noise.Xk
    @test sampled.W==supplied_noise.W
    @test sampled.selected==supplied_noise.selected
    @test sampled.threshold==supplied_noise.threshold
    @test sampled.construction==supplied_noise.construction
    @test rand(sampled_rng)==rand(noise_rng)
  end
  @test all(isequal.(before,(X,y,mu,Sigma,s)))
end

@testset "near-singular covariance remains validated" begin
  Sigma=[1.0 1-1e-8;1-1e-8 1.0]
  # Keep a definite interior margin for this ill-conditioned covariance.
  construction_options=(safety=.1,)
  plan=knockoff_diagonal(Sigma;construction_options)
  @test plan.diagnostics.feasibility==:validated
  @test all(>(0),plan.s)
  D=Diagonal(plan.s)
  @test eigmin(Symmetric([Sigma Sigma-D;Sigma-D Sigma]))>=-100eps(Float64)
  _,A,C=H._knockoff_parameters(Sigma,plan.s)
  @test C'C≈2D-D*A atol=100eps(Float64) rtol=1e-6
  X=randn(Xoshiro(10605),20,2);Z=randn(Xoshiro(10606),20,2)
  K=gaussian_knockoffs(X,zeros(2),Sigma;noise=Z,construction_options)
  @test K≈X-X*A+Z*C
  @test all(isfinite,K)
end

# Diagnostic only: retain visibility of the existing rounded-Cholesky limitation
# without making acceptance of a singular covariance a regression requirement.
let Sigma=fill(6f0,2,2)
  outcome=try
    gaussian_knockoffs(ones(Float32,6,2),zeros(Float32,2),Sigma;noise=zeros(Float32,6,2))
    :accepted
  catch err
    nameof(typeof(err))
  end
  @info "Singular Float32 Gaussian covariance probe" rank=rank(Sigma) rounded_cholesky=isposdef(Symmetric(Sigma)) outcome
end
end
