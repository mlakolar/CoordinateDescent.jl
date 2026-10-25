include("solvers.jl")
using Test
function splitpath(X,y,ls,tau,w,strategy;limit=4.)
 n,p=size(X);start=time();setup=0.;optimize=0.;extract=0.;iters=0;answers=Any[];status="complete"
 m=direct_model(HiGHS.Optimizer());set_silent(m)
 for (name,val) in (("threads",1),("parallel","off"),("solver","simplex"),("simplex_strategy",strategy==:primal ? 4 : 1),("primal_feasibility_tolerance",1e-7),("dual_feasibility_tolerance",1e-7))
  set_optimizer_attribute(m,name,val)
 end
 @variable(m,bp[1:p]>=0);@variable(m,bm[1:p]>=0);@variable(m,a)
 @variable(m,rp[1:n]>=0);@variable(m,rn[1:n]>=0)
 @constraint(m,eq,X*bp .- X*bm .+ a .+ rp .- rn .== y)
 @objective(m,Min,(tau*sum(rp)+(1-tau)*sum(rn))/n+first(ls)*sum(w.*(bp.+bm)))
 vars=vcat(bp,bm);weights=vcat(w,w);setup=time()-start
 for lambda in ls
  time()-start>=limit && (status="budget";break)
  t=time();set_objective_coefficient(m,vars,lambda.*weights);setup+=time()-t
  set_optimizer_attribute(m,"time_limit",max(.01,limit-(time()-start)))
  t=time();optimize!(m);optimize+=time()-t
  termination_status(m)!=MOI.OPTIMAL && (status=string(termination_status(m));break)
  iters+=MOI.get(backend(m),MOI.SimplexIterations())
  t=time();b=value.(bp).-value.(bm);ai=value(a);u=n.*dual.(eq);extract+=time()-t
  push!(answers,(;b,a=ai,u,lambda))
 end
 (;answers,status,setup,optimize,extract,iters,total=time()-start)
end
function data(case,seed)
 n,p=1000,Dict("small"=>100,"wide"=>1000,"ultrawide"=>10000)[case]
 rng=Xoshiro(seed);X=randn(rng,n,p)
 for j in 2:p
  @views X[:,j] .= 0.5 .* X[:,j-1] .+ sqrt(.75) .* X[:,j]
 end
 X,.7 .+ X[:,1:10]*collect(range(1.,.3,length=10)) .+ randn(rng,n)
end
function main()
 out=ARGS[1];mkpath(out);rng=Xoshiro(1010);tx=randn(rng,60,12);ty=tx[:,1:3]*ones(3)+randn(rng,60)
 @testset "split coefficient LP certificates" begin
  for tau in (.2,.5,.8),strategy in (:primal,:dual)
   w=collect(range(1.,2.,length=12));r=splitpath(tx,ty,[.2,.1],tau,w,strategy;limit=10.)
   @test r.status=="complete"
   @test all(-1e-7<=c.gap<=1e-6 for c in (certificate(tx,ty,s.b,s.a,s.lambda,tau,w,s.u) for s in r.answers))
  end
 end
 lppath(tx,ty,[.2,.1],.5,ones(12),:dual,:dual,true;limit=10.)
 for case in ("small","wide","ultrawide"),seed in (1001,1002,1003)
  X,y=data(case,seed);n,p=size(X);w=ones(p);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/n
  ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
  open(joinpath(out,"$case-$seed-paths.csv"),"w") do io
   println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host")
   open(joinpath(out,"$case-$seed-solutions.csv"),"w") do so
    println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
    for rep in 1:2,name in shuffle(Xoshiro(seed+rep),["lp-split-primal-warm","lp-split-dual-warm","lp-dual-dual-warm"])
     GC.gc();timed=@timed name=="lp-dual-dual-warm" ? lppath(X,y,ls,.5,w,:dual,:dual,true;limit=4.) : splitpath(X,y,ls,.5,w,name=="lp-split-primal-warm" ? :primal : :dual;limit=4.)
     r=timed.value
     println(io,join((case,seed,rep,name,n,p,.5,r.status,length(r.answers),timed.time,r.setup,r.optimize,r.extract,timed.bytes,timed.gctime,r.iters,gethostname()),','));flush(io)
     for (k,s) in enumerate(r.answers)
      c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
      println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,NaN,count(abs.(s.b).>1e-7)),','));flush(so)
     end
     println("DONE $case $seed $rep $name $(r.status) $(timed.time)s");flush(stdout)
    end
   end
  end
 end
end
main()
