using Random, LinearAlgebra, Statistics

@testset "Automatic tuning" begin
  X=randn(Xoshiro(910),80,12); y=X[:,1]+randn(Xoshiro(911),80)
  for family in (:gaussian,:binomial,:poisson)
    b=family==:gaussian ? y : family==:binomial ? Float64.(y .> 0) : Float64.(abs.(round.(y)))
    f=debiased_lasso(X,b;family,rng=Xoshiro(912),threads=false)
    g=debiased_lasso(X,b;family,rng=Xoshiro(912),threads=false)
    @test f.converged
    @test f.lambda>0
    @test f.beta == g.beta
    println("AUTO_FIT ",family," lambda=",f.lambda," node=",first(f.projection.lambda))
  end
  a=nodewise_lasso(X;targets=[1,2],rng=Xoshiro(914),threads=false)
  b=nodewise_lasso(X;targets=[4],rng=Xoshiro(914),threads=false)
  @test first(a.lambda)==first(b.lambda)
  c=nodewise_lasso(X;targets=[1,2],rng=Xoshiro(914),threads=false,gram=false)
  @test a.theta≈c.theta atol=1e-5
  @test a.lambda≈c.lambda
  @test debiased_lasso(X,y;initial_tuning=:scaled,nodewise_lambda=:universal).converged
  @test_throws ArgumentError debiased_lasso(X,y;initial_tuning=:bad)
  @test_throws ArgumentError nodewise_lasso(X,:bad)
  @test_throws ArgumentError nodewise_lasso(X;nodewise_cv_nodes=0)
  @test_throws ArgumentError nodewise_lasso(X;nodewise_cv_folds=81)
end


@testset "inference target subsets keep predictor random streams" begin
  rg = Xoshiro(6601)
  X = randn(rg, 80, 24)
  y = 0.2 .+ 0.8 .* X[:,1] .- 0.5 .* X[:,2] .+ randn(rg,80)
  seed = 7741
  full = debiased_lasso(X,y; threads=false,rng=Xoshiro(seed))
  for targets in ([3,1,12], [12,3,1])
    sub = debiased_lasso(X,y; targets,threads=false,rng=Xoshiro(seed))
    @test sub.converged && full.converged
    @test sub.beta ≈ full.beta[targets] atol=1e-11 rtol=1e-11
    @test sub.se ≈ full.se[targets] atol=1e-11 rtol=1e-11
    @test sub.projection.lambda ≈ full.projection.lambda[targets] atol=1e-12 rtol=1e-12
  end
  if Threads.nthreads() > 1
    targets = [12,3,1]
    sub = debiased_lasso(X,y; targets,threads=true,rng=Xoshiro(seed))
    @test sub.beta ≈ full.beta[targets] atol=1e-11 rtol=1e-11
    @test sub.se ≈ full.se[targets] atol=1e-11 rtol=1e-11
  end
end
