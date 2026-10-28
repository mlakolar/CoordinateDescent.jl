using Test,Random,LinearAlgebra,SparseArrays,HighDimensionalModels
const HM100=HighDimensionalModels
@testset "masked signal operations and zero-width band factor" begin
  for T in (Float32,Float64)
    train=[1,3,6];A=HM100._MaskedSignalDesign{T}(7,train);M=Matrix{T}(I,7,7)[train,:]
    b=T[.4,-.3,.2];β=T.(1:7);v=zeros(T,3);w=zeros(T,7)
    @test A*β==M*β
    @test mul!(v,A,β)==M*β
    @test mul!(w,transpose(A),b)==M'b
    @test HM100._structured_gram(A)==M'M
    @test HM100._structured_tproduct(A,b)==M'b
    F=HM100._structured_band_factor(spdiagm(0=>T[1,2,4]),0)
    @test HM100._structured_solve!(zeros(T,3),F,T[2,4,8])≈T[2,2,2]
  end
end
@testset "streamed masked CV agrees with explicit row-selection folds" begin
 rng=Xoshiro(10026);n=30;y=randn(rng,n);folds=repeat(1:3,10);Isp=spdiagm(0=>ones(n));ls=[.02,.005,0.]
 for order in 0:2, irregular in (false,true)
  x=irregular ? cumsum(.7 .+ rand(rng,n)) : nothing
  kwargs=(;order,x,folds,threads=false,abstol=1e-10,reltol=1e-9,max_iter=100000)
  a=cv_trend_filter(y,ls;kwargs...);b=cv_trend_filter(Isp,y,ls;kwargs...)
  @test a.cvm≈b.cvm atol=2e-5 rtol=2e-5
  @test a.cvse≈b.cvse atol=2e-5 rtol=2e-5
  @test a.λmin==b.λmin
  @test a.λ1se==b.λ1se
  @test coef(a.path)≈coef(b.path) atol=2e-5 rtol=2e-5
  @test a.path.family==:trend
 end
 D=difference_matrix(n)
 for fit in ((args...;kw...)->CVGeneralizedLassoPath(args...,D,ls;kw...),)
  a=fit(y;folds,threads=false);b=fit(Isp,y;folds,threads=false)
  @test a.cvm≈b.cvm atol=1e-5
  @test a.cvse≈b.cvse atol=1e-5
 end
 for ratio in (0.,.5,1.)
  a=cv_fused_lasso(y,ls;folds,threads=false,l1_ratio=ratio,abstol=1e-10,reltol=1e-9)
  b=cv_fused_lasso(Isp,y,ls;folds,threads=false,l1_ratio=ratio,abstol=1e-10,reltol=1e-9)
  @test a.cvm≈b.cvm atol=2e-5 rtol=2e-5
  @test a.cvse≈b.cvse atol=2e-5 rtol=2e-5
 end
end
@testset "masked fold leakage, identification, and zero penalty" begin
 n=18;train=[1,4,7,10,13,16];test=setdiff(1:n,train);y=sin.(collect(1.:n));D=difference_matrix(n,2);K=difference_matrix(n)
 A=HM100._MaskedSignalDesign{Float64}(n,train)
 collectfits(b)=begin
   out=Vector{Vector{Float64}}()
   HM100._signal_admm_errors(A,b,D,K,[.01,.003],(β,j,ok)->(ok || error("not converged");push!(out,copy(β)));family=:trend,trend_order=1,abstol=1e-10,reltol=1e-9,max_iter=100000)
   out
 end
 changed=copy(y);changed[test].+=1000
 @test collectfits(y[train])==collectfits(changed[train])
 zeroβ=Ref{Any}(nothing)
 HM100._signal_admm_errors(A,y[train],D,K,[0.],(β,j,ok)->(zeroβ[]=β))
 @test zeroβ[][train]==y[train]
 @test iszero(zeroβ[][test])
 @test_throws ArgumentError cv_trend_filter(y,[.01];order=2,folds=[ones(Int,n-1);2],threads=false)
 @test_throws ArgumentError cv_trend_filter(y,[.01];max_concurrent_folds=0)
 @test_throws ErrorException cv_trend_filter(y,[.01];max_iter=0,threads=false)
 @test_throws ArgumentError cv_trend_filter(y,[.01];intercept=true,threads=false)
end
@testset "masked blocked-fold and bounded threaded agreement" begin
 y=sin.(collect(1.:40));folds=repeat(1:4;inner=10);ls=[.02,.005]
 a=cv_trend_filter(y,ls;folds,threads=false,order=1,max_iter=100000)
 b=cv_trend_filter(y,ls;folds,threads=true,max_concurrent_folds=2,order=1,max_iter=100000)
 c=cv_trend_filter(y,ls;folds,threads=true,max_concurrent_folds=1,order=1,max_iter=100000)
 @test a.cvm==b.cvm==c.cvm
 @test a.cvse==b.cvse==c.cvse
 @test a.folds==b.folds==c.folds==folds
end
