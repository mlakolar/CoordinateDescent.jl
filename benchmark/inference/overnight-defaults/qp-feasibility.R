.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3',.libPaths()))
suppressPackageStartupMessages(library(CVXR));out=list();at=0
for(rep in 1:3) {
 X=as.matrix(read.csv(file.path('outputs','development','gaussian-correlated',rep,'X.csv'),header=FALSE));X=scale(X,center=TRUE,scale=FALSE);X=sweep(X,2,sqrt(colMeans(X^2)),'/');G=crossprod(X)/nrow(X);p=ncol(X);n=nrow(X);j=5
 sv=svd(X,nu=0,nv=min(n,p));V=sv$v[,sv$d>max(sv$d)*1e-10,drop=FALSE]
 for(solver in intersect(c('CLARABEL','ECOS','SCS'),installed_solvers())) {
  theta=Variable(p);t=Variable(1);e=rep(0,p);e[j]=1;constraints=list(G%*%theta-e<=t,e-G%*%theta<=t,t>=0);prob=Problem(Minimize(t),constraints)
  res=if(solver=='SCS') solve(prob,solver=solver,eps=1e-7,max_iters=30000) else solve(prob,solver=solver)
  a=as.vector(res$getDualValue(constraints[[1]]));b=as.vector(res$getDualValue(constraints[[2]]));u=b-a;u=u-as.vector(V%*%crossprod(V,u));u=u/max(1,sum(abs(u)));if(u[j]<0)u=-u
  lower=u[j];station=max(abs(G%*%u));mu=.5*sqrt(2*log(p)/n);at=at+1
  out[[at]]=data.frame(rep=rep,target=j,solver=solver,status=res$status,primal_value=res$value,actual_residual=max(abs(G%*%res$getValue(theta)-e)),dual_lower_bound=lower,dual_null_residual=station,dual_l1=sum(abs(u)),half_mu=mu,half_mu_below_numerical_lower_bound=lower>mu+1e-6&&station<1e-8)
 }
}
write.csv(do.call(rbind,out),'qp-feasibility-multisolver.csv',row.names=FALSE);cat('QP_FEASIBILITY_DONE',at,'\n')
