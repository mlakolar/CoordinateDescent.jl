# Native-sslasso-inspired data-only convergence search, with strict certificates.
# Distinct from original sslasso: CV initializer, tighter solve, robust HC1 SE.
include("score-core.jl")
family==:gaussian || error("Gaussian adaptive QP only")
qpopt=CDOptions(;opt_tol=1e-8,gaptol=0,max_iter=5000)
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".qpadapt.complete") && continue
 X,y,b,targets=dataset(rep)
 open(stem*".qpadapt.csv","w") do io
  println(io,header)
  open(stem*".qpadapt.diagnostics.csv","w") do di
   println(di,"policy,target,selected_mu,attempted_levels,failed_mu,projection_error,passes,moment_error,derivative_error,shift")
   try
    started=time_ns();rng=Xoshiro(970000+rep)
    cv=CVLassoPath(X,y,opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng)
    base=lasso(X,y,cv.λmin,opt;intercept=true,standardize_x=true,rng)
    sparsefit=lasso(X,y,cv.λ1se,opt;intercept=true,standardize_x=true,rng=Xoshiro(970000+rep))
    base.converged&&sparsefit.converged || error("initial CV lasso not converged")
    bb,bb0,status,ok=postfit(X,y,Vector(sparsefit.x),sparsefit.intercept,rep)
    for j in targets
     global targets_current=[j]
     try
      proj=nothing;failed=NaN;attempts=0
      # Stop at first uncertified level. This is a reproducible solver rule,
      # not a claim that all smaller constraints are mathematically infeasible.
      for level in sqrt(2log(p)/n) .* (0.8 .^ (0:10))
       attempts+=1
       try
        trial=debiasing_matrix(X;method=:qp,mu=level,targets=[j],threads=false,options=qpopt,rng=Xoshiro(993000+rep+j))
        proj=trial
       catch e
        failed=level
        write(stem*".target"*string(j)*".qpadapt.search-warning",sprint(showerror,e))
        break
       end
      end
      proj===nothing && error("no certified projection at starting constraint")
      for (policy,init,intercept,lambda,st) in [("adaptive_cvmin",Vector(base.x),base.intercept,cv.λmin,"regularized"),("adaptive_cv1se_refit",bb,bb0,cv.λ1se,status)]
       f=debiased_lasso(X,y,lambda;family,targets=[j],initial=init,initial_intercept=intercept,projection=proj,robust=true,options=opt)
       f.converged || error("uncertified final fit")
       df=n-count(!iszero,init)-1;df>0 || error("no residual degrees of freedom")
       emit(io,policy*"_one_step_hc1",j,b,f.beta[1],f.se[1]*sqrt(n/df),f,(time_ns()-started)/1e9,st)
       est,se,rse,ge,de,it,shift=rootfit(X,y,f,1)
       emit(io,policy*"_score_hc1",j,b,est,rse*sqrt(n/df),f,(time_ns()-started)/1e9,st)
       println(di,join((policy,j,proj.lambda[1],attempts,failed,proj.error[1],proj.iterations[1],ge,de,shift),','))
      end
     catch e
      write(stem*".target"*string(j)*".qpadapt.error",sprint(showerror,e,catch_backtrace()))
     end
    end
   catch e
    write(stem*".initializer.qpadapt.error",sprint(showerror,e,catch_backtrace()))
   end
  end
 end
 write(stem*".qpadapt.complete","attempted all targets; inspect complete-target validity and search warnings\n")
end
