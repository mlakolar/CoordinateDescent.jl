using Random, LinearAlgebra, Statistics
@testset "LDPE bias and noise tuning" begin
  X=randn(Xoshiro(880),120,9);X[:,2] .= .7X[:,1] .+ .7X[:,2]
  X .*= transpose(collect(1.:9.));X .+= 2
  opt=CDOptions(;opt_tol=1e-10)
  P=nodewise_lasso(X,:ldpe;targets=[1,3],options=opt,threads=false,rng=Xoshiro(881))
  @test all(P.converged)
  @test P.tuning==:ldpe
  @test all(P.bias_target_met)
  @test all(P.noise_factor .<= 1.25P.reference_noise .+ 1e-10)
  @test P.noise_factor≈vec(sqrt.(sum(abs2,P.scores;dims=1)))/size(X,1)
  C=X.-mean(X;dims=1);Z=C./transpose(P.scales)
  @test diag(P.theta*(C'C/120)[:,[1,3]])≈ones(2) atol=1e-7
  for (a,j) in enumerate(P.targets)
    others=setdiff(1:9,j)
    fit=lasso(Z[:,others],Z[:,j],P.lambda[a],opt)
    v=zeros(9);v[j]=1;v[others] .= -Vector(fit.x)
    tau=dot(Z[:,j],Z*v)/120
    @test Vector(P.theta[a,:])≈v./(tau.*P.scales.*P.scales[j]) atol=1e-6
  end
  Q=nodewise_lasso(X,:ldpe;targets=[1,3],options=opt,gram=false,threads=false,rng=Xoshiro(881))
  @test P.theta≈Q.theta atol=1e-6
  D=hcat(X[:,1],X[:,1]);R=nodewise_lasso(D,:ldpe;targets=[1],options=opt,threads=false)
  @test !R.bias_target_met[1]
  @test nodewise_lasso(X[:,1:1],:ldpe;options=opt).lambda==[0.]
  @test_throws ArgumentError nodewise_lasso(X,:ldpe;ldpe_kappa=-1)
end

@testset "CV paths with unreached tail levels" begin
  errors=[2. 1. NaN; 3. 2. NaN; 1. 1. NaN]
  m,se,best,one_se=HighDimensionalModels._cvsummary(errors,ones(3))
  @test m[end]==Inf
  @test best==2
  @test one_se in 1:2
  @test_throws ArgumentError HighDimensionalModels._cvsummary(fill(NaN,3,2),ones(3))
end

@testset "Restricted LDPE projections" begin
  X=randn(Xoshiro(885),100,8)
  X[:,2] .= .8X[:,1] .+ .4X[:,2]
  X[:,3] .= .6X[:,1] .+ .6X[:,3]
  opt=CDOptions(;opt_tol=1e-11)
  P=nodewise_lasso(X,:ldpe;targets=[1],ldpe_unpenalized=2,options=opt,threads=false,rng=Xoshiro(886))
  C=X.-mean(X;dims=1);Z=C./transpose(P.scales)
  c=vec(Z'Z[:,1]);selected=sort!([k for k in 2:8];by=k -> (-abs(c[k]),k))[1:2]
  @test all(P.converged)
  @test maximum(abs,C[:,selected]'P.scores[:,1])<1e-6
  @test dot(C[:,1],P.scores[:,1])/100≈1 atol=1e-7
  weights=ones(7);weights[selected.-1] .= 0
  f=lasso(Z[:,2:8],Z[:,1],P.lambda[1],weights,opt)
  v=vcat(1.,-Vector(f.x));tau=dot(Z[:,1],Z*v)/100
  @test Vector(P.theta[1,:])≈v./(tau.*P.scales.*P.scales[1]) atol=1e-6
  Q=nodewise_lasso(X,:ldpe;targets=[1],ldpe_unpenalized=2,gram=false,options=opt,threads=false,rng=Xoshiro(886))
  @test P.theta≈Q.theta atol=1e-6
  @test_throws ArgumentError nodewise_lasso(X,:cv;ldpe_unpenalized=1)
  @test_throws ArgumentError nodewise_lasso(X,:ldpe;ldpe_unpenalized=8)
end
