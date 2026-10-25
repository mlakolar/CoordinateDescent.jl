using HighDimensionalModels,ProximalBase,Random,LinearAlgebra,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
caseid,shard,maxrep,stage=ARGS;shard=parse(Int,shard);maxrep=parse(Int,maxrep)
C=readdlm("cases.csv",',',Any,'\n';header=true)[1];row=findfirst(==(caseid),C[:,1]);c=C[row,:]
family=Symbol(c[2]);n=Int(c[3]);p=Int(c[4]);rho=Float64(c[5]);s=Int(c[6]);signal=Float64(c[7]);randomsupport=Int(c[8])==1;errtype=c[9];design=c[10];alpha=Float64(c[11]);cy=Float64(c[12])
root=joinpath("outputs",stage,caseid);mkpath(root);opt=CDOptions(;opt_tol=1e-8,gaptol=0)
function dataset(rep)
 rng=Xoshiro((stage=="canary" ? 92000000 : stage=="development" ? 92100000 : 92200000)+row*10000+rep)
 if design=="belloni"
  Z=randn(rng,n,p-1)
  for j=2:p-1;Z[:,j].=rho.*Z[:,j-1].+sqrt(1-rho^2).*Z[:,j];end
  vy=zeros(p);vy[1:5]=1 ./collect(1:5);vy[11:15]=1 ./collect(1:5);vd=zeros(p);vd[1:10]=1 ./collect(1:10)
  d=vd[1].+Z*vd[2:end]+randn(rng,n);X=hcat(d,Z);b=vcat(alpha,cy.*vy[2:end]);b0=cy*vy[1];targets=vcat(1,collect(p-4:p))
 else
  X=randn(rng,n,p)
  for j=2:p;X[:,j].=rho.*X[:,j-1].+sqrt(1-rho^2).*X[:,j];end
  X.-=mean(X;dims=1);X./=sqrt.(mean(abs2,X;dims=1))
  ids=randomsupport ? randperm(rng,p)[1:s] : collect(1:s);b=zeros(p);b[ids].=signal.*(-1.).^(0:s-1);b0=.2
  if design=="sihr";b.=0;b[1]=.5;b[2]=1.;b0=-.5;ids=[1,2];end
  targets=vcat(ids[1:min(s,5)],[j for j=1:p if b[j]==0][1:5])
 end
 eta=b0.+X*b
 if family==:binomial;y=Float64.(rand(rng,n).<1 ./ (1 .+exp.(-eta)))
 elseif errtype=="hetero";y=eta.+sqrt.((1 .+X[:,end].^2)./2).*randn(rng,n)
 elseif errtype=="t3";y=eta.+randn(rng,n)./sqrt.(vec(sum(abs2,randn(rng,n,3);dims=2)))
 else;y=eta.+randn(rng,n);end
 X,y,b,targets
end
function postfit(X,y,b,b0,rep)
 ids=findall(!iszero,b)
 length(ids)<size(X,1)-5 || return b,b0,"fallback_size",true
 if isempty(ids);return zeros(length(b)),family==:gaussian ? mean(y) : log(clamp(mean(y),1e-6,1-1e-6)/(1-clamp(mean(y),1e-6,1-1e-6))),"empty",true;end
 M=hcat(ones(size(X,1)),X[:,ids]);rank(M)==size(M,2) || return b,b0,"fallback_rank",true
 if family==:gaussian;v=M\y;bb=zeros(length(b));bb[ids]=v[2:end];return bb,v[1],"refit",true;end
 f=glm_lasso(X[:,ids],y,0.,ProximalBase.Logistic(),opt;intercept=true,standardize_x=true,rng=Xoshiro(19+rep));v=Vector(f.x)
 if !f.converged || maximum(abs,X[:,ids]*v.+f.intercept)>20;return b,b0,"fallback_separation",true;end
 bb=zeros(length(b));bb[ids]=v;bb,f.intercept,"refit",f.converged
end
function runrep(rep)
 dest=joinpath(root,string(rep));isfile(dest*".jcomplete") && return
 X,y,b,targets=dataset(rep);mkpath(dest)
 for (name,a) in [("X",X),("y",y),("beta",b),("targets",targets)];writedlm(joinpath(dest,name*".csv"),a,',');end
 open(dest*".julia.csv","w") do io
 println(io,"method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter")
 baseline=nothing;bb=nothing;bb0=0.;status="regularized"
 methods=family==:gaussian ? ["rldpe","rldpe_refit","qp","qp_refit","qp_half","qp_half_refit"] : ["rldpe","rldpe_refit"]
 for method in methods
  t=time_ns()
  try
   if method=="rldpe"
    baseline=debiased_lasso(X,y;family,targets,threads=false,options=opt,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,robust=family==:gaussian,rng=Xoshiro(970000+rep));f=baseline
    bb,bb0,status,_=postfit(X,y,f.initial,f.intercept,rep)
   else
    isref=endswith(method,"refit");init=isref ? bb : baseline.initial;init0=isref ? bb0 : baseline.intercept
    policy=startswith(method,"qp") ? (;method=:qp,mu=(occursin("half",method) ? .5 : 1.)*sqrt(2log(p)/n)) : (;nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5)
    f=debiased_lasso(X,y,baseline.lambda;family,targets,initial=init,initial_intercept=init0,threads=false,options=opt,robust=family==:gaussian,policy...,rng=Xoshiro(970000+rep))
   end
   ci=confint(f);elapsed=(time_ns()-t)/1e9
   for a in eachindex(targets);j=targets[a];println(io,join((method,j,b[j],f.beta[a],f.se[a],ci[a,1],ci[a,2],f.pvalues[a],f.converged,elapsed,endswith(method,"refit") ? status : "regularized",count(!iszero,f.initial),f.lambda,maximum(f.projection.lambda)),','));end
  catch e
   open(dest*"."*method*".error","w") do er;println(er,sprint(showerror,e,catch_backtrace()));end
  end
 end
 end
 write(dest*".jcomplete","ok\n")
end
for rep=shard:8:maxrep
 try;runrep(rep);catch e;open(joinpath(root,string(rep)*".dataset.error"),"w") do io;println(io,sprint(showerror,e,catch_backtrace()));end;end
end
