# Standard model-based and HC0/1/2/3 variance comparisons, no truth tuning.
include("score-core.jl")
family==:gaussian || error("Gaussian variance study only")
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".gvar.complete") && continue
 X,y,b,targets=dataset(rep)
 open(stem*".gvar.csv","w") do io
  println(io,header)
  open(stem*".gvar.diagnostics.csv","w") do di
   println(di,"policy,initial_size,max_leverage,min_leverage,mean_leverage,leverage_trace_error,projection_converged")
   try
    start=time_ns()
    baseline=debiased_lasso(X,y;family,targets,threads=false,options=opt,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,robust=false,rng=Xoshiro(970000+rep))
    baseline.converged || error("initial/projection not converged")
    for policy in ["gvar_cvmin","gvar_cv1se_refit"]
     try
      if policy=="gvar_cvmin";f=baseline;status="regularized"
      else
       cv=CVLassoPath(X,y,opt;nfolds=5,nλ=40,intercept=true,standardize_x=true,threads=false,rng=Xoshiro(970000+rep))
       initialfit=lasso(X,y,cv.λ1se,opt;intercept=true,standardize_x=true,rng=Xoshiro(970000+rep))
       initialfit.converged || error("1SE initial did not converge")
       bb,bb0,status,ok=postfit(X,y,Vector(initialfit.x),initialfit.intercept,rep)
       f=debiased_lasso(X,y,cv.λ1se;family,targets,initial=bb,initial_intercept=bb0,projection=baseline.projection,options=opt,robust=false)
      end
      selected=findall(!iszero,f.initial);M=hcat(ones(n),X[:,selected]);k=size(M,2)
      rank(M)==k && k<n || error("selected design rank/size invalid")
      Q=Matrix(qr(M).Q)[:,1:k];leverage=vec(sum(abs2,Q;dims=2))
      traceerror=abs(sum(leverage)-k);traceerror<1e-8 || error("leverage trace mismatch")
      minimum(leverage)>=0 && maximum(leverage)<1-1e-8 || error("invalid leverage")
      # Diagonal projection identity supplies an independent leverage check.
      direct=vec(sum((M/(M'*M)).*M;dims=2))
      maximum(abs,direct-leverage)<1e-8 || error("QR/direct leverage mismatch")
      println(di,join((policy,length(selected),maximum(leverage),minimum(leverage),mean(leverage),traceerror,all(f.projection.converged)),','))
      for (a,j) in enumerate(targets)
       emit(io,policy*"_model",j,b,f.beta[a],f.se[a],f,(time_ns()-start)/1e9,status)
       for hc in ["hc0","hc1","hc2","hc3"]
        corrected=hc=="hc2" ? f.residuals./sqrt.(1 .-leverage) : hc=="hc3" ? f.residuals./(1 .-leverage) : f.residuals
        influence=f.projection.scores[:,a].*corrected;influence.-=mean(influence)
        se=norm(influence)/n*(hc=="hc1" ? sqrt(n/(n-k)) : 1.)
        emit(io,policy*"_"*hc,j,b,f.beta[a],se,f,(time_ns()-start)/1e9,status)
       end
      end
     catch e
      write(stem*"."*policy*".gvar.error",sprint(showerror,e,catch_backtrace()))
     end
    end
   catch e
    write(stem*".initializer.gvar.error",sprint(showerror,e,catch_backtrace()))
   end
  end
 end
 write(stem*".gvar.complete","attempted both initializers and standard variance rules\n")
end
