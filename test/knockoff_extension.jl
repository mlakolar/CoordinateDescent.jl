module KnockoffExtensionTests
using HighDimensionalModels, LinearAlgebra, Random, Test
import Knockoffs
const H=HighDimensionalModels
@test Base.get_extension(H,:HighDimensionalModelsKnockoffsExt)!==nothing

function joint_objective(R,a,strategy)
  D=Diagonal(a);G=Symmetric([R R-D;R-D R])
  strategy===:mvr ? sum(inv,eigvals(G)) : -logdet(G)
end

@testset "optional diagonal optimizer and native sampling" begin
  rng=Xoshiro(106)
  for T in (Float32,Float64),rho in (0.,.4,.8),units in (1e-20,1.,1e20),strategy in (:mvr,:maxent)
    p=4;n=24;scales=T.(sqrt(units).*[.2,.7,2,5])
    R0=[rho^abs(i-j) for i in 1:p,j in 1:p]
    S=T.(R0.*(scales*scales'));before=copy(S)
    plan=knockoff_diagonal(S;strategy)
    @test plan.strategy==strategy && plan.diagnostics.backend==:Knockoffs
    @test plan.diagnostics.backend_status==:unavailable
    @test plan.diagnostics.backend_iterations===nothing
    @test plan.diagnostics.stationarity_passed
    @test plan.diagnostics.feasibility==:validated
    @test eltype(plan.s)==T && S==before
    d=Float64.(diag(S));R=Float64.(S)./sqrt.(d*d');a=Float64.(plan.s)./d
    @test eigmin(Symmetric([R R-Diagonal(a);R-Diagonal(a) R]))>0
    objective=joint_objective(R,a,strategy)
    @test objective≈plan.diagnostics.objective rtol=1e-6 atol=1e-6
    # Full 2p-by-2p objective finite differences are independent of the
    # production partitioned inverse/gradient certificate implementation.
    h=2e-6
    for j in 1:p
      left=copy(a);right=copy(a);left[j]-=h;right[j]+=h
      gradient=(joint_objective(R,right,strategy)-joint_objective(R,left,strategy))/(2h)
      @test abs(gradient)/max(1,abs(objective))<5e-4
    end
    X=T.(randn(rng,n,p).*scales');mu=zeros(T,p);Z=T.(randn(rng,n,p))
    X0,Z0=copy(X),copy(Z)
    K=gaussian_knockoffs(X,mu,S;strategy,noise=Z)
    @test K==gaussian_knockoffs(X,mu,S;s=plan.s,noise=Z)
    @test X==X0 && Z==Z0 && S==before
    _,A,C=H._knockoff_parameters(S,plan.s)
    @test Float64.(S)*Float64.(A)≈Matrix(Diagonal(Float64.(plan.s))) rtol=500eps(T)
    target=2Diagonal(Float64.(plan.s))-Diagonal(Float64.(plan.s))*(Float64.(S)\Matrix(Diagonal(Float64.(plan.s))))
    @test Float64.(C)'Float64.(C)≈target rtol=1000eps(T) atol=1000eps(T)*units
  end
  X=randn(rng,40,5);y=randn(rng,40);S=[.4^abs(i-j) for i in 1:5,j in 1:5];mu=zeros(5);Z=randn(rng,40,5)
  for strategy in (:mvr,:maxent)
    K=fixed_knockoffs(X;strategy);d=knockoff_diagonal(X'X;strategy).s
    @test K'K≈X'X atol=1e-9
    @test X'K≈X'X-Diagonal(d) atol=1e-9
    result=knockoff_filter(X,y,.2;Sigma=S,mu,noise=Z,strategy,order=:cyclic,opt_tol=1e-9)
    @test result.construction.strategy==strategy && result.construction.diagnostics.stationarity_passed
    @test result.Xk==gaussian_knockoffs(X,mu,S;strategy,noise=Z)
    # A deliberately inadequate iteration cap must not silently fall back.
    hard=[.7^abs(i-j) for i in 1:12,j in 1:12]
    @test_throws Exception knockoff_diagonal(hard;strategy,construction_options=(niter=1,optimality_tol=1e-12))
  end
end
end
