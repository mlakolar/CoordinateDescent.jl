# The fully solved QP score is a distinct estimator, not standard JM one-step.
include("score-core.jl")
family==:gaussian || error("QP moment study is Gaussian only")
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".qpscore.complete") && continue
 X,y,b,targets=dataset(rep);global targets_current=targets
 open(stem*".qpscore.csv","w") do io
  println(io,header)
  open(stem*".qpscore.diagnostics.csv","w") do di
   println(di,"method,target,moment_error,derivative_error,iterations,shift,initial_converged,projection_converged")
   try
    start=time_ns();rng=Xoshiro(970000+rep)
    cv=CVLassoPath(X,y,opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng)
    initialfit=lasso(X,y,cv.λmin,opt;intercept=true,standardize_x=true,rng)
    initialfit.converged || error("CVmin initial did not converge")
    proj=debiasing_matrix(X;method=:qp,mu=sqrt(2log(p)/n),targets,options=opt,threads=false,intercept=true,standardize_x=true,rng)
    all(proj.converged) || error("QP projection not certified")
    for policy in ["qp_cvmin","qp_cvmin_refit","qp_cv1se_refit"]
     try
      if policy=="qp_cvmin"
       bb=Vector(initialfit.x);bb0=initialfit.intercept;status="regularized";lambda=cv.λmin
      else
       fit=policy=="qp_cvmin_refit" ? initialfit : lasso(X,y,cv.λ1se,opt;intercept=true,standardize_x=true,rng=Xoshiro(970000+rep))
       fit.converged || error("initial did not converge")
       bb,bb0,status,ok=postfit(X,y,Vector(fit.x),fit.intercept,rep)
       lambda=policy=="qp_cvmin_refit" ? cv.λmin : cv.λ1se
      end
      f=debiased_lasso(X,y,lambda;family,targets,initial=bb,initial_intercept=bb0,projection=proj,options=opt,robust=true)
      f.converged || error("uncertified QP fit")
      df=n-count(!iszero,bb)-1;df>0 || error("no residual degrees of freedom")
      for (a,j) in enumerate(targets)
       elapsed=(time_ns()-start)/1e9
       emit(io,policy*"_one_step",j,b,f.beta[a],f.se[a],f,elapsed,status)
       emit(io,policy*"_one_step_hc1",j,b,f.beta[a],f.se[a]*sqrt(n/df),f,elapsed,status)
       try
        est,se,rse,ge,de,it,shift=rootfit(X,y,f,a)
        emit(io,policy*"_score",j,b,est,se,f,(time_ns()-start)/1e9,status)
        emit(io,policy*"_score_hc1",j,b,est,rse*sqrt(n/df),f,(time_ns()-start)/1e9,status)
        println(di,join((policy,j,ge,de,it,shift,f.converged,all(proj.converged)),','))
       catch e
        write(stem*"."*policy*".target"*string(j)*".qpscore.error",sprint(showerror,e,catch_backtrace()))
       end
      end
     catch e
      write(stem*"."*policy*".qpscore.error",sprint(showerror,e,catch_backtrace()))
     end
    end
   catch e
    write(stem*".initializer.qpscore.error",sprint(showerror,e,catch_backtrace()))
   end
  end
 end
 write(stem*".qpscore.complete","attempted all QP policies; inspect complete target sets and errors\n")
end
