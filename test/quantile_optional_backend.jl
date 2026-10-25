# A fresh process verifies installation without loading the optional backend.
using HighDimensionalModels, Random, Test
@testset "quantile API without HiGHS loaded" begin
  @test Base.get_extension(HighDimensionalModels,:HighDimensionalModelsHiGHSExt) === nothing
  X=reshape([1.,2.,3.],:,1);y=[1.,1.5,3.]
  @test isfinite(quantile_lambda(X;nsim=20,rng=Xoshiro(7)))
  err=try
    quantile_lasso(X,y,.1)
  catch e
    e
  end
  @test err isa ArgumentError
  @test occursin("using HiGHS",sprint(showerror,err))
end
