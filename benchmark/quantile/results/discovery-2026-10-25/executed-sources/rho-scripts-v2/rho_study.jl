include("solvers.jl")
include("certificate_admm.jl")
include("adaptive_admm.jl")
using Test
function data(case,seed)
    n,p=1000,case=="small" ? 100 : 1000
    rng=Xoshiro(seed);X=randn(rng,n,p)
    for j in 2:p
        @views X[:,j] .= 0.5 .* X[:,j-1] .+ sqrt(.75) .* X[:,j]
    end
    X,.7 .+ X[:,1:10]*collect(range(1.,.3,length=10)) .+ randn(rng,n)
end
function rhosolve(name,rho,X,y,ls)
    w=ones(size(X,2))
    name=="adaptive" ? adaptiveadmmpath(X,y,ls,.5,w;rho,limit=30.) : gapadmmpath(X,y,ls,.5,w;rho,limit=30.)
end
function main()
 out=ARGS[1];mkpath(out)
 rng=Xoshiro(900);tx=randn(rng,60,12);ty=tx[:,1:3]*ones(3)+randn(rng,60)
 t=adaptiveadmmpath(tx,ty,[.2,.1],.5,ones(12);limit=10.)
 lppath(tx,ty,[.2,.1],.5,ones(12),:dual,:dual,true;limit=10.)
 @test t.status=="complete"
 @test all(-1e-7<=certificate(tx,ty,s.b,s.a,s.lambda,.5,ones(12),s.u).gap<=1e-5*(1+objective(tx,ty,s.b,s.a,s.lambda,.5,ones(12))) for s in t.answers)
 open(joinpath(out,"tuning.csv"),"w") do pilot
  println(pilot,"case,method,rho,status,levels,total_s,maxgap")
  for case in ("small","wide")
   X,y=data(case,901);p=size(X,2);w=ones(p);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/1000
   ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
   choices=[("fixed",r) for r in (.01,.1,1.,10.)];push!(choices,("adaptive",1.))
   best=nothing;besttime=Inf
   for (name,rho) in shuffle(Xoshiro(901),choices)
    GC.gc();r=rhosolve(name,rho,X,y,ls)
    gaps=[certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u).gap for s in r.answers]
    ok=length(r.answers)==8 && all(-1e-7<=c.gap<=1e-5*(1+abs(c.primal)) for c in (certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u) for s in r.answers))
    println(pilot,join((case,name,rho,r.status,length(r.answers),r.total,maximum(gaps;init=-Inf)),','));flush(pilot)
    ok && r.total<besttime && (best=(name,rho);besttime=r.total)
    println("TUNING $case $name rho=$rho $(r.status) $(r.total)s");flush(stdout)
   end
   best===nothing && continue
   open(joinpath(out,"$case-choice.txt"),"w") do io;println(io,"$(best[1]) rho=$(best[2]); selected on seed901 only; confirmation seeds902,903,904");end
   for seed in (902,903,904)
    X,y=data(case,seed);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/1000
    ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
    open(joinpath(out,"$case-$seed-paths.csv"),"w") do io
     println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host")
     open(joinpath(out,"$case-$seed-solutions.csv"),"w") do so
      println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
      for rep in 1:2, name in shuffle(Xoshiro(seed+rep),["tuned-admm","lp-dual-dual-warm"])
       GC.gc();timed=@timed name=="tuned-admm" ? rhosolve(best[1],best[2],X,y,ls) : lppath(X,y,ls,.5,w,:dual,:dual,true;limit=30.)
       r=timed.value
       println(io,join((case,seed,rep,name,1000,p,.5,r.status,length(r.answers),timed.time,r.setup,r.optimize,r.extract,timed.bytes,timed.gctime,r.iters,gethostname()),','));flush(io)
       for (k,s) in enumerate(r.answers)
        c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
        println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,NaN,count(abs.(s.b).>1e-7)),','));flush(so)
       end
       println("CONFIRM $case $seed $rep $name $(r.status) $(timed.time)s");flush(stdout)
      end
     end
    end
   end
  end
 end
end
main()
