using SHA
# Frozen Gaussian candidate. No tuning against confirmation outcomes.
include("confirmation-data.jl")
family==:gaussian || error("Frozen Gaussian rule only")
for rep=shard:8:maxrep
 stem=joinpath(root,string(rep));isfile(stem*".gaussian.complete") && continue
 X,y,b,targets=dataset(rep)
 try
  started=time_ns()
  f=debiased_lasso(X,y;family,targets,threads=false,rng=Xoshiro(970000+rep))
  elapsed=(time_ns()-started)/1e9
  f.converged && f.variance==:hc3 && f.projection.tuning==:ldpe || error("unexpected/nonconverged production policy")
  se=f.se;ids=findall(!iszero,f.initial);M=hcat(ones(n),X[:,ids]);k=size(M,2)
  rank(M)==k && k<n || error("selected design rank/size failure")
  Q=Matrix(qr(M).Q)[:,1:k];h=vec(sum(abs2,Q;dims=2))
  ratio=f.projection.noise_factor./f.projection.reference_noise
  all(isfinite,ratio) && maximum(ratio)<=1.5*(1+1e-8) || error("LDPE SD budget failure")
  xhash=bytes2hex(SHA.sha256(reinterpret(UInt8,vec(X))));yhash=bytes2hex(SHA.sha256(reinterpret(UInt8,y)))
  open(stem*".gaussian.csv","w") do io
   println(io,"method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter")
   for (a,j) in enumerate(targets)
    est=f.beta[a];pv=2*HighDimensionalModels._pnorm(-abs(est/se[a]))
    println(io,join(("gaussian_default_hc3",j,b[j],est,se[a],est-1.959963984540054*se[a],est+1.959963984540054*se[a],pv,true,elapsed,"regularized_frozen",length(ids),f.lambda,f.projection.lambda[a]),','))
   end
  end
  open(stem*".gaussian.diagnostics.csv","w") do io
   println(io,"target,initial_size,sigma,max_leverage,leverage_trace_error,projection_error,bias_target_met,path_boundary,noise_ratio,projection_passes,input_x_sha256,input_y_sha256")
   for (a,j) in enumerate(targets)
    println(io,join((j,length(ids),f.sigma,maximum(h),abs(sum(h)-k),f.projection.error[a],f.projection.bias_target_met[a],f.projection.path_boundary[a],ratio[a],f.projection.iterations[a],xhash,yhash),','))
   end
  end

 catch e
  write(stem*".gaussian.error",sprint(showerror,e,catch_backtrace()))
 end
 write(stem*".gaussian.complete","attempted frozen policy; inspect validity and error receipts\n")
end
