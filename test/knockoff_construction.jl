module KnockoffConstructionTests
using HighDimensionalModels, LinearAlgebra, Random, Test
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
  for strategy in (:mvr,:maxent),opts in ((niter=0,),(tol=0.,),(λmin=-1.,),(robust=1,),(optimality_tol=0.,),(unknown=1,))
    @test_throws ArgumentError knockoff_diagonal(S;strategy,construction_options=opts)
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
end
end
