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
