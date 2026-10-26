using HighDimensionalModels,Test,Random,LinearAlgebra,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
@testset "20 replicated target-set and thread invariance checks" begin
 for rep=1:20
  rg=Xoshiro(880000+rep);X=randn(rg,80,24);y=0.2 .+ 0.8 .* X[:,1] .- 0.5 .* X[:,2] .+ randn(rg,80)
  full=debiased_lasso(X,y;threads=false,rng=Xoshiro(551+rep))
  @test full.converged
  for ts in ([3,1,12],[12,3,1])
   sub=debiased_lasso(X,y;targets=ts,threads=false,rng=Xoshiro(551+rep))
   @test sub.converged
   @test sub.beta ≈ full.beta[ts] atol=1e-11 rtol=1e-11
   @test sub.se ≈ full.se[ts] atol=1e-11 rtol=1e-11
   @test sub.projection.lambda ≈ full.projection.lambda[ts] atol=1e-12 rtol=1e-12
  end
  ts=[12,3,1];threaded=debiased_lasso(X,y;targets=ts,threads=true,rng=Xoshiro(551+rep))
  @test threaded.beta ≈ full.beta[ts] atol=1e-11 rtol=1e-11
  @test threaded.se ≈ full.se[ts] atol=1e-11 rtol=1e-11
 end
end
# Reuse existing meaningful correctness checks; no performance claims from tests.
for name in ("inference","inference_golden","inference_tuning","inference_ldpe")
 include(joinpath(@__DIR__,"..","package-hc3-v2","test",name*".jl"))
end
