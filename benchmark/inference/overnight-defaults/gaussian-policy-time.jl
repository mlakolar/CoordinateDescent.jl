using HighDimensionalModels,Random,LinearAlgebra,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
engine,caseid,rs=ARGS;rep=parse(Int,rs);base=joinpath("competitor-data",caseid,rs)
X=readdlm(joinpath(base,"X.csv"),',',Float64);y=vec(readdlm(joinpath(base,"y.csv"),',',Float64));targets=Int.(vec(readdlm(joinpath(base,"targets.csv"),',',Float64)));truth=vec(readdlm(joinpath(base,"truth.csv"),',',Float64))
out=joinpath("outputs","gaussian-native",caseid,rs*"."*engine);mkpath(dirname(out))
try
 if engine=="ours_warm"
  rg=Xoshiro(177);debiased_lasso(randn(rg,80,12),randn(rg,80);targets=1:2,threads=false,rng=Xoshiro(77));GC.gc()
 end
 t=time_ns();f=debiased_lasso(X,y;targets,threads=false,rng=Xoshiro(970000+rep));elapsed=(time_ns()-t)/1e9
 f.converged && f.variance==:hc3 || error("nonconverged or incorrect production recipe")
 open(out*".csv","w") do io
  println(io,"method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter")
  for (a,j) in enumerate(targets)
   est=f.beta[a];s=f.se[a];pv=2*HighDimensionalModels._pnorm(-abs(est/s))
   println(io,join((engine,j,truth[j],est,s,est-1.959963984540054*s,est+1.959963984540054*s,pv,true,elapsed,"production_automatic",count(!iszero,f.initial),f.lambda,f.projection.lambda[a]),','))
  end
 end
 open(out*".meta","w") do io
  println(io,"status=ok\nfit_seconds=",elapsed,"\nactual_inferred_targets=",length(targets),"\nphase=",engine)
  if engine=="ours_warm"
   # Separate replicated component measurement, not a subtraction-based decomposition.
   opt=CDOptions(;opt_tol=1e-8,gaptol=0);rg=Xoshiro(970000+rep)
   t=time_ns();cv=CVLassoPath(X,y,opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng=rg);tcv=(time_ns()-t)/1e9
   t=time_ns();initial=lasso(X,y,cv.λmin,opt;intercept=true,standardize_x=true,rng=rg);tinit=(time_ns()-t)/1e9
   maximum(abs,Vector(initial.x)-f.initial)<1e-8 && abs(cv.λmin-f.lambda)<1e-12 || error("initializer component parity")
   println(io,"separate_initial_cv_seconds=",tcv,"\nseparate_initial_refit_seconds=",tinit)
  end
 end
catch e
 write(out*".error",sprint(showerror,e,catch_backtrace()))
end
write(out*".complete","attempt complete; inspect errors/status\n")
