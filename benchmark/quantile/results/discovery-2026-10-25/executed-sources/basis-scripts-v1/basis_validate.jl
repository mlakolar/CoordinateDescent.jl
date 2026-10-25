include("solvers.jl")
include("null_basis.jl")
using Test
@testset "analytic quantile basis" begin
 rng=Xoshiro(1100)
 for n in (60,61),tau in (.2,.5,.8)
  X=randn(rng,n,12);y=.7 .+ X[:,1:3]*ones(3) .+ randn(rng,n)
  r=nullpath(X,y,[10.,.2,.05],tau,ones(12);limit=10.)
  @test r.status=="complete"
  @test first(r.answers).iterations==0
  for s in r.answers
   c=certificate(X,y,s.b,s.a,s.lambda,tau,ones(12),s.u)
   @test -1e-7<=c.gap<=1e-6
  end
 end
end
