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
 if family==:gaussian
  ids=findall(!iszero,f.initial);M=hcat(ones(n),X[:,ids]);k=size(M,2)
  rank(M)==k && k<n || error("selected-design rank/size failure")
  Q=Matrix(qr(M).Q)[:,1:k];lever=vec(sum(abs2,Q;dims=2))
  abs(sum(lever)-k)<1e-8 && maximum(lever)<1 || error("leverage check failed")
  residual=residual./(1 .-lever)
 end
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
