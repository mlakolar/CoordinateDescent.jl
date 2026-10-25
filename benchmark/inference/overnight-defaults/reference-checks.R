.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(hdm);library(CVXR)})
writeLines(capture.output(sessionInfo()),'reference-session.txt')
cases=read.csv('cases.csv');records=list();checks=list();counter=0
for(caseid in cases$case[cases$family=='gaussian']) {
 stem=file.path('outputs','canary',caseid,1);X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));y=scan(file.path(stem,'y.csv'),quiet=TRUE);ts=scan(file.path(stem,'targets.csv'),quiet=TRUE);truth=scan(file.path(stem,'beta.csv'),quiet=TRUE)
 for(j in ts) {
  f=hdm::rlassoEffect(X[,-j,drop=FALSE],y,X[,j,drop=FALSE],method='double selection',post=TRUE)
  ids=if(is.null(f$selection.index)) integer(0) else which(as.logical(f$selection.index));N=X[,-j,drop=FALSE];M=cbind(1,X[,j],N[,ids,drop=FALSE]);b=qr.solve(M,y);r=as.vector(y-M%*%b)
  v=as.vector(lm.fit(cbind(1,N[,ids,drop=FALSE]),X[,j])$residuals);n=nrow(X);k=length(ids)
  se=sqrt(sum(v^2*r^2)/sum(v^2)^2*n/(n-k-1))
  counter=counter+1;checks[[counter]]=data.frame(case=caseid,target=j,selected=k,coefficient_difference=abs(unname(f$alpha)-b[2]),se_difference=abs(unname(f$se)-se))
  records[[counter]]=data.frame(case=caseid,rep=1,method='hdm_ds_single',target=j,truth=truth[j],estimate=unname(f$alpha),se=unname(f$se),lower=unname(f$alpha)-qnorm(.975)*unname(f$se),upper=unname(f$alpha)+qnorm(.975)*unname(f$se),pvalue=unname(f$pval))
 }
}
write.csv(do.call(rbind,checks),'reference-hdm-checks.csv',row.names=FALSE);write.csv(do.call(rbind,records),'reference-hdm-records.csv',row.names=FALSE)
stopifnot(max(vapply(checks,function(x)x$coefficient_difference,numeric(1)))<1e-8,max(vapply(checks,function(x)x$se_difference,numeric(1)))<1e-8)
# Feasibility diagnostic for the half-universal QP. This is numerical, not an exact rational certificate.
feas=list()
for(caseid in c('gaussian-moderate','gaussian-correlated')) {
 stem=file.path('outputs','canary',caseid,1);X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));X=scale(X,center=TRUE,scale=FALSE);X=sweep(X,2,sqrt(colMeans(X^2)),'/');G=crossprod(X)/nrow(X);p=ncol(X);j=5
 theta=Variable(p);t=Variable(1);e=rep(0,p);e[j]=1
 prob=Problem(Minimize(t),list(G%*%theta-e<=t,e-G%*%theta<=t,t>=0))
 solvers=installed_solvers();solver=if('CLARABEL'%in%solvers)'CLARABEL' else if('ECOS'%in%solvers)'ECOS' else 'SCS'
 result=solve(prob,solver=solver);v=result$getValue(theta)
 feas[[caseid]]=data.frame(case=caseid,solver=solver,status=result$status,minimax_objective=result$value,actual_residual=max(abs(G%*%v-e)),universal=sqrt(2*log(p)/nrow(X)),half_universal=.5*sqrt(2*log(p)/nrow(X)))
}
write.csv(do.call(rbind,feas),'reference-qp-feasibility.csv',row.names=FALSE)
cat('REFERENCE_CHECKS_DONE',counter,'\n')
