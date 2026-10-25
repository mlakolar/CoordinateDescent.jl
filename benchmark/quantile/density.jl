include("solvers.jl")
using Printf
function main()
 out=ARGS[1];mkpath(out)
 open(joinpath(out,"density.csv"),"w") do io
  println(io,"seed,rep,n,p,levels,method,total_s,pivots,status,maxgap,host")
  for seed in (401,402,403)
   rng=Xoshiro(seed); X=randn(rng,1000,100); y=.7 .+ X[:,1:10]*ones(10) .+ randn(rng,1000)
   w=ones(100); tau=.5; lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/1000
   methods=["highs-primal","highs-dual","gurobi-primal","gurobi-dual"]
   for name in methods
    engine,strategy=Symbol.(split(name,'-'))
    lppath(X[1:40,1:10],y[1:40],[.1,.05],tau,ones(10),:primal,strategy,true;engine)
   end
   for rep in 1:3, (levels,name) in shuffle(rng,collect(Iterators.product((8,32,128),methods)))
    engine,strategy=Symbol.(split(name,'-')); ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=levels)))
    GC.gc();r=lppath(X,y,ls,tau,w,:primal,strategy,true;engine,limit=30.)
    gap=maximum(certificate(X,y,s.b,s.a,s.lambda,tau,w,s.u).gap for s in r.answers;init=-Inf)
    println(io,join((seed,rep,1000,100,levels,name,r.total,r.iters,r.status,gap,gethostname()),','));flush(io)
   end
  end
 end
end
main()
