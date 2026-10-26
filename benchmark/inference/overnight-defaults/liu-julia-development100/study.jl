using HighDimensionalModels,ProximalBase,DelimitedFiles,Random,LinearAlgebra
BLAS.set_num_threads(1)
shard=parse(Int,ARGS[1]);root="outputs/liu-julia-development100";mkpath(root)
for rep=shard:8:100
 stem=joinpath(root,string(rep));isfile(stem*".complete")&&continue
 source=joinpath("outputs/development/liu-published",string(rep));X=readdlm(joinpath(source,"cal-X.csv"),',',Float64);y=vec(readdlm(joinpath(source,"cal-y.csv"),',',Float64));truth=vec(readdlm(joinpath(source,"cal-truth.csv"),',',Float64));ts=Int.(vec(readdlm(joinpath(source,"cal-ts.csv"),',',Float64)))[[1,end]]
 open(stem*".csv","w") do io
  println(io,"method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds")
  for policy in ["julia_logistic_default","julia_logistic_restricted_expert"]
   try
    kw=policy=="julia_logistic_default" ? (;) : (;nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,options=CDOptions(;opt_tol=1e-8,gaptol=0))
    t=time_ns();f=debiased_lasso(X,y;family=:binomial,targets=ts,threads=false,rng=Xoshiro(970000+rep),kw...);elapsed=(time_ns()-t)/1e9
    f.converged||error("inference not converged")
    for (a,j) in enumerate(ts)
     b=f.beta[a];s=f.se[a];isfinite(b)&&isfinite(s)&&s>0||error("invalid inference")
     pv=2*HighDimensionalModels._pnorm(-abs(b/s));abs(pv-f.pvalues[a])<1e-12||error("pvalue mismatch")
     println(io,join((policy,j,truth[j],b,s,b-1.959963984540054*s,b+1.959963984540054*s,pv,true,elapsed),','))
    end
   catch e
    write(stem*"."*policy*".error",sprint(showerror,e,catch_backtrace()))
   end
  end
 end
 write(stem*".complete","attempted both frozen prior recipes; inspect errors\n")
end
