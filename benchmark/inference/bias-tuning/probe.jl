using CoordinateDescent, ProximalBase, Random, LinearAlgebra, Statistics
BLAS.set_num_threads(1)
stage,case,shard,maxrep=ARGS
shard=parse(Int,shard);maxrep=parse(Int,maxrep)
family=startswith(case,"gaussian") ? :gaussian : :binomial
kind=last(split(case,'-'))
n,p,rho,s=kind=="correlated" ? (100,500,.9,15) : kind=="benign" ? (500,100,.3,5) : (200,500,.3,5)
base=stage=="development" ? 91500000 : 91600000
caseoffset=kind=="correlated" ? 10000 : kind=="benign" ? 20000 : kind=="random" ? 30000 : 0
outdir=joinpath("probe",stage,case);mkpath(outdir)
opt=CDOptions(;opt_tol=1e-8)
function data(rep)
 rg=Xoshiro(base+caseoffset+rep);X=randn(rg,n,p)
 for j=2:p;X[:,j] .= rho.*X[:,j-1] .+ sqrt(1-rho^2).*X[:,j];end
 X .-= mean(X;dims=1);X ./=sqrt.(mean(abs2,X;dims=1))
 ids=kind=="random" ? randperm(rg,p)[1:s] : collect(1:s)
 beta=zeros(p);beta[ids] .= (family==:gaussian ? .5 : .7).*(-1.).^(0:s-1)
 eta=.2 .+ X*beta
 y=family==:gaussian ? eta+randn(rg,n) : Float64.(rand(rg,n).<1 ./(1 .+exp.(-eta)))
 null=kind=="random" ? [j for j in 1:p if beta[j]==0][1:5] : collect(s+1:s+5)
 X,y,beta,vcat(ids[1:5],null)
end
function refit(X,y,b,b0,rep)
 ids=findall(!iszero,b)
 length(ids)>=n÷3 && return b,b0,true,"fallback_size"
 isempty(ids) && return b,b0,true,"empty_support"
 if family==:gaussian
  M=hcat(ones(n),X[:,ids]);rank(M)==size(M,2) || return b,b0,true,"fallback_rank"
  v=M\y;bb=zeros(p);bb[ids]=v[2:end]
  return bb,v[1],true,"refitted"
 else
  f=glm_lasso(X[:,ids],y,0.,ProximalBase.Logistic(),opt;intercept=true,standardize_x=true,rng=Xoshiro(930000+rep))
  if !f.converged || maximum(abs,X[:,ids]*Vector(f.x).+f.intercept)>15
   return b,b0,true,"fallback_glm"
  end
  bb=zeros(p);bb[ids]=Vector(f.x)
  return bb,f.intercept,true,"refitted"
 end
end
function emit(io,co,rep,method,f,truth,elapsed,initstatus,initialok=true)
 ci=confint(f);act=truth.!=0;null=.!act
 cv=(ci[:,1].<=truth).&(truth.<=ci[:,2]);rej=f.pvalues.<.05
 P=f.projection
 for j in eachindex(truth)
  println(co,join((rep,method,P.targets[j],truth[j],f.beta[j],f.se[j],f.pvalues[j],P.lambda[j],P.bias_factor[j],P.noise_factor[j],P.reference_noise[j],P.bias_target_met[j],P.path_boundary[j]),','))
 end
 println(io,join((rep,method,f.converged&&initialok,mean(cv[act]),mean(cv[null]),mean(rej[null]),mean(rej[act]),mean(ci[act,2]-ci[act,1]),mean(ci[null,2]-ci[null,1]),mean((f.beta[act]-truth[act]).*sign.(truth[act])./f.se[act]),f.sigma,f.lambda,minimum(P.lambda),maximum(P.lambda),maximum(P.bias_factor),maximum(P.noise_factor./P.reference_noise),all(P.bias_target_met),count(!iszero,f.initial),norm(f.initial),elapsed,initstatus),','))
 flush(io);flush(co)
end
function runrep(rep)
 dest=joinpath(outdir,string(rep));isfile(dest*".complete") && return
 X,y,beta,targets=data(rep)
 open(dest*".csv","w") do io
  open(dest*".coefficients","w") do co
   println(io,"rep,method,converged,coverage_active,coverage_null,type1,power,length_active,length_null,signed_z_bias_active,sigma,lambda,node_lambda_min,node_lambda_max,bias_factor_max,noise_ratio_max,bias_target_met,initial_size,initial_norm,seconds,initial_status")
   println(co,"rep,method,target,truth,estimate,se,pvalue,node_lambda,bias_factor,noise_factor,reference_noise,bias_target_met,path_boundary")
   t=time_ns();f=debiased_lasso(X,y;family,targets,threads=false,options=opt,rng=Xoshiro(940000+rep))
   emit(io,co,rep,"cv",f,beta[targets],(time_ns()-t)/1e9,"regularized")
   t=time_ns();g=debiased_lasso(X,y,f.lambda;family,initial=f.initial,initial_intercept=f.intercept,nodewise_lambda=:ldpe,targets,threads=false,options=opt,rng=Xoshiro(940000+rep))
   emit(io,co,rep,"ldpe",g,beta[targets],(time_ns()-t)/1e9,"regularized")
   b,b0,ok,status=refit(X,y,f.initial,f.intercept,rep)
   for (method,bb,bb0) in [("ldpe_refit",b,b0),("ldpe_relaxed",.5b+.5f.initial,.5b0+.5f.intercept)]
    t=time_ns();kw=family==:gaussian ? (;projection=g.projection) : (;nodewise_lambda=:ldpe,targets,threads=false,rng=Xoshiro(940000+rep))
    r=debiased_lasso(X,y,f.lambda;family,initial=bb,initial_intercept=bb0,options=opt,kw...)
    emit(io,co,rep,method,r,beta[targets],(time_ns()-t)/1e9,status,ok)
   end
   t=time_ns()
   half=family==:gaussian ? lasso(X,y,.5f.lambda,opt;intercept=true,standardize_x=true,rng=Xoshiro(950000+rep)) : glm_lasso(X,y,.5f.lambda,ProximalBase.Logistic(),opt;intercept=true,standardize_x=true,rng=Xoshiro(950000+rep))
   kw=family==:gaussian ? (;projection=g.projection) : (;nodewise_lambda=:ldpe,targets,threads=false,rng=Xoshiro(940000+rep))
   r=debiased_lasso(X,y,.5f.lambda;family,initial=Vector(half.x),initial_intercept=half.intercept,options=opt,kw...)
   emit(io,co,rep,"ldpe_half",r,beta[targets],(time_ns()-t)/1e9,"regularized",half.converged)
  end
 end
 write(dest*".complete","ok\n")
end
# Compile on independent small inputs; actual per-method costs are recorded,
# and total process cost is available from the Slurm step/job log.
W=randn(Xoshiro(12),80,8);yy=family==:gaussian ? W[:,1]+randn(Xoshiro(13),80) : Float64.(rand(Xoshiro(13),80) .< 1 ./ (1 .+ exp.(-W[:,1])))
debiased_lasso(W,yy;family,nodewise_lambda=:ldpe,threads=false,rng=Xoshiro(14))
for rep=shard:8:maxrep
 try
  runrep(rep)
 catch e
  dest=joinpath(outdir,string(rep));open(dest*".error","w") do io;println(io,sprint(showerror,e,catch_backtrace()));end
 end
end
println("PROBE_DONE ",stage," ",case," shard=",shard)
