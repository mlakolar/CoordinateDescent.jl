# Development-only fully solved target/intercept moments. Not a production default.
include("score-data.jl")
sigmoid(v)=map(t->t>=0 ? inv(1+exp(-t)) : exp(t)/(1+exp(t)),v)
function rootfit(X,y,f,a)
 n=size(X,1);j=targets_current[a];d=X[:,j];eta=X*f.initial.+f.intercept
 if family==:gaussian
  z=f.projection.scores[:,a]
 else
  u=sigmoid(eta);h=u.*(1 .-u);center=vec(X'*h)/sum(h)
  z=(X.-center')*f.projection.theta[a,:]
 end
 z=z/sqrt(mean(abs2,z));G=hcat(ones(n),z);D=hcat(ones(n),d)
 offset=eta.-f.intercept.-d.*f.initial[j];theta=[f.intercept,f.initial[j]]
 moments(v)=begin
  linear=offset+D*v;m=family==:gaussian ? linear : sigmoid(linear)
  h=family==:gaussian ? ones(n) : m.*(1 .-m)
  G'*(y-m)/n,G'*(h.*D)/n,m,h
 end
 iterations=0
 for k=1:60
  g,J,m,h=moments(theta);iterations=k
  maximum(abs,g)<1e-10 && break
  cond(J)<1e10 || error("ill-conditioned score Jacobian")
  delta=J\g;step=min(1.,2/max(norm(delta),2.));accepted=false
  for ls=1:25
   trial=theta+step*delta;gn,_,_,_=moments(trial)
   if norm(gn)<norm(g)
    theta=trial;accepted=true;break
   end
   step/=2
  end
  accepted || error("score line search failed")
 end
 g,J,m,h=moments(theta)
 maximum(abs,g)<1e-8 || error("score root not converged")
 J[2,2]-J[2,1]*J[1,2]/J[1,1]>1e-8 || error("nonpositive target score derivative")
 all(isfinite,theta) && minimum(h)>0 || error("nonfinite or separated score root")
 # Independent finite-difference derivative check of the two empirical moments.
 fd=hcat([(moments(theta+1e-5*[q==v for q=1:2])[1]-moments(theta-1e-5*[q==v for q=1:2])[1])/2e-5 for v=1:2]...)
 derivative_error=maximum(abs,fd+J)/max(1.,maximum(abs,J))
 derivative_error<1e-6 || error("score derivative finite-difference mismatch")
 if family==:gaussian
  direct=(G'*D)\(G'*(y-offset))
  maximum(abs,direct-theta)<1e-7 || error("Gaussian direct moment solution mismatch")
 end
 influence=(G*inv(J)')[:,2]
 residual=y-m
 se=family==:gaussian ? sqrt(sum(abs2,influence.*residual.-mean(influence.*residual)))/n : sqrt(sum(influence.^2 .*h))/n
 robustse=sqrt(sum(abs2,influence.*residual.-mean(influence.*residual)))/n
 theta[2],se,robustse,maximum(abs,g),derivative_error,iterations,theta[2]-f.initial[j]
end
header="method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter"
function emit(io,method,j,b,est,se,f,seconds,status)
 isfinite(se)&&se>0 || error("invalid score SE")
 pv=2*HighDimensionalModels._pnorm(-abs(est/se))
 println(io,join((method,j,b[j],est,se,est-1.959963984540054*se,est+1.959963984540054*se,pv,f.converged,seconds,status,count(!iszero,f.initial),f.lambda,maximum(f.projection.lambda)),','))
end
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".score.complete") && continue
 X,y,b,targets=dataset(rep);global targets_current=targets
 open(stem*".score.csv","w") do io
  println(io,header)
  open(stem*".score.diagnostics.csv","w") do di
   println(di,"method,target,moment_error,derivative_error,iterations,shift,initial_converged,projection_converged")
   start=time_ns();baseline=nothing
   for policy in ["cvmin","relaxed_half","cv1se_refit"]
    try
     if policy=="cvmin"
      baseline=debiased_lasso(X,y;family,targets,threads=false,options=opt,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,robust=family==:gaussian,rng=Xoshiro(970000+rep));f=baseline;status="regularized"
     elseif policy=="relaxed_half"
      bb,bb0,status,ok=postfit(X,y,baseline.initial,baseline.intercept,rep)
      f=debiased_lasso(X,y,baseline.lambda;family,targets,initial=(baseline.initial+bb)/2,initial_intercept=(baseline.intercept+bb0)/2,threads=false,options=opt,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,robust=family==:gaussian,rng=Xoshiro(970000+rep))
     else
      cv=family==:gaussian ? CVLassoPath(X,y,opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng=Xoshiro(970000+rep)) : CVGLMLassoPath(X,y,ProximalBase.Logistic(),opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng=Xoshiro(970000+rep))
      initialfit=family==:gaussian ? lasso(X,y,cv.λ1se,opt;intercept=true,standardize_x=true,rng=Xoshiro(970000+rep)) : glm_lasso(X,y,cv.λ1se,ProximalBase.Logistic(),opt;intercept=true,standardize_x=true,rng=Xoshiro(970000+rep))
      initialfit.converged || error("one-SE initializer did not converge")
      bb,bb0,status,ok=postfit(X,y,Vector(initialfit.x),initialfit.intercept,rep)
      f=debiased_lasso(X,y,cv.λ1se;family,targets,initial=bb,initial_intercept=bb0,threads=false,options=opt,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,robust=family==:gaussian,rng=Xoshiro(970000+rep))
     end
     f.converged || error("initializer or projection not converged")
     elapsed=(time_ns()-start)/1e9
     for (a,j) in enumerate(targets)
      emit(io,policy*"_one_step",j,b,f.beta[a],f.se[a],f,elapsed,status)
      if family==:gaussian
       emit(io,policy*"_one_step_hc1",j,b,f.beta[a],f.se[a]*sqrt(n/(n-count(!iszero,f.initial)-1)),f,elapsed,status)
      end
     end
     # Root errors are target-specific, and must count against method eligibility.
     for (a,j) in enumerate(targets)
      try
       est,se,rse,ge,de,it,shift=rootfit(X,y,f,a)
       elapsed=(time_ns()-start)/1e9
       emit(io,policy*"_score",j,b,est,se,f,elapsed,status)
       emit(io,policy*"_score_hc1",j,b,est,rse*sqrt(n/(n-count(!iszero,f.initial)-1)),f,elapsed,status)
       println(di,join((policy,j,ge,de,it,shift,f.converged,all(f.projection.converged)),','))
      catch e
       write(stem*"."*policy*".target"*string(j)*".score.error",sprint(showerror,e,catch_backtrace()))
      end
     end
    catch e
     write(stem*"."*policy*".score.error",sprint(showerror,e,catch_backtrace()))
    end
   end
  end
 end
 write(stem*".score.complete","attempted all policies; inspect errors and per-target counts\n")
end
