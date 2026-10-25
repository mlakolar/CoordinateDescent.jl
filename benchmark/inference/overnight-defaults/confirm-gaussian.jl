# Frozen Gaussian candidate. No tuning against confirmation outcomes.
include("confirmation-data.jl")
family==:gaussian || error("Frozen Gaussian rule only")
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".confirm.complete") && continue
 X,y,b,targets=dataset(rep)
 try
  started=time_ns()
  f=debiased_lasso(X,y;family,targets,threads=false,options=opt,
   initial_tuning=:cv,initial_cv_folds=5,initial_cv_nlambda=40,
   intercept=true,standardize_x=true,nodewise_lambda=:ldpe,
   ldpe_unpenalized=4,ldpe_kappa=.5,ldpe_nlambda=60,ldpe_minratio=.001,
   ldpe_bias_target=sqrt(2log(max(p,2))),ldpe_relax=.1,
   robust=false,rng=Xoshiro(970000+rep))
  f.converged || error("initial/projection convergence failure")
  ids=findall(!iszero,f.initial);M=hcat(ones(n),X[:,ids]);k=size(M,2)
  rank(M)==k && k<n || error("selected design rank/size failure")
  Q=Matrix(qr(M).Q)[:,1:k];h=vec(sum(abs2,Q;dims=2))
  abs(sum(h)-k)<1e-8 || error("leverage trace failure")
  maximum(abs,vec(sum((M/(M'*M)).*M;dims=2))-h)<1e-8 || error("QR/direct leverage mismatch")
  minimum(h)>=0 && maximum(h)<1-1e-8 || error("leverage outside valid range")
  ratio=f.projection.noise_factor./f.projection.reference_noise
  all(isfinite,ratio) && maximum(ratio)<=1.5*(1+1e-8) || error("LDPE SD-budget failure")
  influence=f.projection.scores.*(f.residuals./(1 .-h))
  influence.-=mean(influence;dims=1);se=vec(sqrt.(sum(abs2,influence;dims=1)))/n
  all(x->isfinite(x)&&x>0,se) || error("invalid HC3 standard error")
  elapsed=(time_ns()-started)/1e9
  open(stem*".confirm.csv","w") do io
   println(io,"method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter")
   for (a,j) in enumerate(targets)
    est=f.beta[a];pv=2*HighDimensionalModels._pnorm(-abs(est/se[a]))
    println(io,join(("frozen_gaussian_hc3",j,b[j],est,se[a],est-1.959963984540054*se[a],est+1.959963984540054*se[a],pv,true,elapsed,"regularized_frozen",length(ids),f.lambda,f.projection.lambda[a]),','))
   end
  end
  open(stem*".confirm.diagnostics.csv","w") do io
   println(io,"target,initial_size,sigma,max_leverage,leverage_trace_error,projection_error,bias_target_met,path_boundary,noise_ratio,projection_passes")
   for (a,j) in enumerate(targets)
    println(io,join((j,length(ids),f.sigma,maximum(h),abs(sum(h)-k),f.projection.error[a],f.projection.bias_target_met[a],f.projection.path_boundary[a],ratio[a],f.projection.iterations[a]),','))
   end
  end
  if stage=="confirmation-canary"
   writedlm(stem*".selected-design.csv",M,',')
   writedlm(stem*".hc3-moments.csv",hcat(f.residuals,h,f.projection.scores),',')
  end
 catch e
  write(stem*".confirm.error",sprint(showerror,e,catch_backtrace()))
 end
 write(stem*".confirm.complete","attempted frozen policy; inspect validity and error receipts\n")
end
