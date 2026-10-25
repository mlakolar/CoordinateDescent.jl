# Development-only variance audit. Native selection/estimation are unchanged.
args=commandArgs(TRUE);caseid=args[1];shard=as.integer(args[2]);maxrep=as.integer(args[3]);stage=args[4]
.libPaths(c('/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages(library(hdm))
cases=read.csv('cases.csv');stopifnot(cases$family[match(caseid,cases$case)]=='binomial')
native=hdm::rlassologitEffect;instrumented=native;bd=body(instrumented)
stopifnot(identical(bd[[length(bd)]],quote(return(results))))
bd[[length(bd)]]=quote(return(list(native=results,selected=I,fit=l3,z=z,S21=S21,S22=S22)))
body(instrumented)=bd
for(rep in seq(shard,maxrep,by=8)) {
 stem=file.path('outputs',stage,caseid,rep);if(file.exists(paste0(stem,'.hdmvar.complete'))) next
 if(!file.exists(paste0(stem,'.jcomplete'))) stop('saved input dataset not ready')
 X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));y=scan(file.path(stem,'y.csv'),quiet=TRUE);truth=scan(file.path(stem,'beta.csv'),quiet=TRUE);ts=scan(file.path(stem,'targets.csv'),quiet=TRUE)
 out=paste0(stem,'.hdmvar.csv');diagout=paste0(stem,'.hdmvar.diagnostics.csv')
 writeLines('method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter',out)
 writeLines('target,selected_size,rank,model_columns,native_S21,native_S22,full_model_S22,coefficient_parity,se_parity,converged,extreme_probability',diagout)
 for(j in ts) {
  warnings=character();started=proc.time()[[3]]
  tryCatch(withCallingHandlers({
   set.seed(980000+rep);f=instrumented(X[,-j,drop=FALSE],y,X[,j],post=TRUE)
   fit=f$fit;M=model.matrix(fit);n=nrow(X);k=ncol(M);h=fit$fitted.values*(1-fit$fitted.values)
   if(!fit$converged||fit$rank!=k||k>=n-5||any(!is.finite(fit$coefficients))||any(h<=0)) stop('invalid selected logistic model')
   if(any(grepl('did not converge|not converge|numerically 0 or 1',warnings,ignore.case=TRUE))) stop('source convergence/separation warning')
   Hinv=solve(crossprod(M,M*as.vector(h)));res=y-fit$fitted.values
   modelSE=sqrt(Hinv[2,2]);sandwich=Hinv%*%crossprod(M,M*as.vector(res^2))%*%Hinv
   hc1SE=sqrt(sandwich[2,2]*n/(n-k));maxSE=sqrt(max(f$S21/n,Hinv[2,2]))
   # The instrumentation must not change the original estimator or variance.
   coefpar=separ=NA_real_
   if(rep==1) {
    set.seed(980000+rep);original=native(X[,-j,drop=FALSE],y,X[,j],post=TRUE)
    coefpar=abs(as.numeric(original$alpha)-as.numeric(f$native$alpha));separ=abs(as.numeric(original$se)-as.numeric(f$native$se))
    if(coefpar>1e-10||separ>1e-10) stop('native instrumentation parity failed')
   }
   beta=as.numeric(f$native$alpha);stopifnot(abs(beta-fit$coefficients[2])<1e-10)
   for(method in c('hdm_glm_model','hdm_glm_hc1','hdm_max_fullmodel')) {
    se=switch(method,hdm_glm_model=modelSE,hdm_glm_hc1=hc1SE,hdm_max_fullmodel=maxSE)
    if(!is.finite(se)||se<=0) stop('invalid variance')
    row=data.frame(method=method,target=j,truth=truth[j],estimate=beta,se=se,lower=beta-qnorm(.975)*se,upper=beta+qnorm(.975)*se,pvalue=2*pnorm(-abs(beta/se)),converged=TRUE,seconds=proc.time()[[3]]-started,refit_status='native_selection_refit_variance_audit',initial_size=k-1,lambda=NA,projection_parameter=NA)
    write.table(row,out,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE)
   }
   dd=data.frame(target=j,selected_size=sum(f$selected),rank=fit$rank,model_columns=k,native_S21=f$S21,native_S22=f$S22,full_model_S22=n*Hinv[2,2],coefficient_parity=coefpar,se_parity=separ,converged=fit$converged,extreme_probability=min(fit$fitted.values,1-fit$fitted.values))
   write.table(dd,diagout,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE)
  },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),paste0(stem,'.target',j,'.hdmvar.error')))
  if(length(warnings))writeLines(warnings,paste0(stem,'.target',j,'.hdmvar.warnings'))
 }
 writeLines('attempted all targets; inspect complete-target validity',paste0(stem,'.hdmvar.complete'))
}
