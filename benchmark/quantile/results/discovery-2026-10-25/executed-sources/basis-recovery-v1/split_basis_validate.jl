include("solvers.jl")
include("initialized_split.jl")
using Test
@testset "split quantile null basis" begin
 rng=Xoshiro(1200)
 for n in (60,61),tau in (.2,.5,.8),strategy in (:primal,:dual)
  X=randn(rng,n,12);y=.7 .+ X[:,1:3]*ones(3) .+ randn(rng,n)
  r=initializedsplitpath(X,y,[10.,.2,.05],tau,ones(12),strategy;limit=10.)
  @test r.status=="complete"
  @test first(r.answers).iterations==0
  for s in r.answers
   c=certificate(X,y,s.b,s.a,s.lambda,tau,ones(12),s.u)
   @test -1e-7<=c.gap<=1e-6
  end
 end
end
