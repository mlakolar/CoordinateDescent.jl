# Refresh current Julia defaults on the existing paired competitor pilot inputs.
using HighDimensionalModels, Random, LinearAlgebra, Statistics, DelimitedFiles
BLAS.set_num_threads(1)
case,shard=ARGS;shard=parse(Int,shard)
family=startswith(case,"gaussian") ? :gaussian : :binomial
prior="/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/data"
outdir=joinpath("default-refresh",case);mkpath(outdir)
policies=family==:gaussian ? [("ours","default",(;)),("oursqp","default",(;method=:qp)),("oursrldpe","experimental",(;nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5))] : [("ours","default",(;)),("oursrldpe","experimental",(;nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5))]
W=randn(Xoshiro(177),80,12);yy=family==:gaussian ? randn(Xoshiro(178),80) : Float64.(rand(Xoshiro(178),80).<.5)
for (_,_,policy) in policies
 debiased_lasso(W,yy;family,targets=[1,2],threads=false,rng=Xoshiro(770001),policy...)
end
for rep=shard:8:20
 dest=joinpath(outdir,string(rep));isfile(dest*".complete") && continue
 dir=joinpath(prior,case,string(rep));X=readdlm(joinpath(dir,"X.csv"),',',Float64)
 y=vec(readdlm(joinpath(dir,"y.csv"),',',Float64));beta=vec(readdlm(joinpath(dir,"beta.csv"),',',Float64));targets=vec(readdlm(joinpath(dir,"targets.csv"),Int));truth=beta[targets];active=truth.!=0;null=.!active
 open(dest*".csv","w") do io
  println(io,"scenario,rep,engine,mode,status,active_coverage,null_coverage,null_rejection,active_rejection,active_length,null_length,fit_seconds,lambda,node_lambda,sigma,actual_inferred_targets")
  open(dest*".coefficients","w") do co
   println(co,"scenario,rep,engine,mode,target,truth,estimate,se,lower,upper,pvalue")
   for (engine,mode,policy) in policies
    try
     GC.gc();t=time_ns();f=debiased_lasso(X,y;family,targets,threads=false,rng=Xoshiro(770000+rep),policy...);seconds=(time_ns()-t)/1e9
     ci=confint(f);covered=(ci[:,1].<=truth).&(truth.<=ci[:,2]);rejected=f.pvalues.<.05
     println(io,join((case,rep,engine,mode,f.converged ? "ok" : "nonconverged",mean(covered[active]),mean(covered[null]),mean(rejected[null]),mean(rejected[active]),mean(ci[active,2]-ci[active,1]),mean(ci[null,2]-ci[null,1]),seconds,f.lambda,first(f.projection.lambda),f.sigma,length(targets)),','))
     for a in eachindex(targets);println(co,join((case,rep,engine,mode,targets[a],truth[a],f.beta[a],f.se[a],ci[a,1],ci[a,2],f.pvalues[a]),','));end
    catch e
     write(dest*"-"*engine*".error",sprint(showerror,e,catch_backtrace()))
     println(io,join((case,rep,engine,mode,"error",fill("NaN",10)...,length(targets)),','))
    end
    flush(io);flush(co)
   end
  end
 end
 write(dest*".complete","processed\n")
end
println("DEFAULT_REFRESH_DONE ",case," shard=",shard)
