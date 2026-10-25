args=commandArgs(TRUE);caseid=args[1];shard=as.integer(args[2]);maxrep=as.integer(args[3]);stage=args[4]
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(hdm);library(glmnet)})
source('../Xia-functions.R')
cases=read.csv('cases.csv');family=cases$family[match(caseid,cases$case)];root=file.path('outputs',stage,caseid)
for(rep in seq(shard,maxrep,by=8)) {
 stem=file.path(root,rep);if(file.exists(paste0(stem,'.supp.complete'))) next
 if(!file.exists(paste0(stem,'.jcomplete'))) stop('missing frozen input data')
 X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));y=scan(file.path(stem,'y.csv'),quiet=TRUE);truth=scan(file.path(stem,'beta.csv'),quiet=TRUE);ts=scan(file.path(stem,'targets.csv'),quiet=TRUE)
 out=paste0(stem,'.supp.csv');writeLines('method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter',out)
 methods=if(family=='gaussian') 'hdm_ds_single' else if(nrow(X)>ncol(X)+5) c('xia_refined','xia_refined_1se') else character()
 for(method in methods) {
  warnings=character();started=proc.time()[[3]]
  tryCatch(withCallingHandlers({
   set.seed(980000+rep)
   if(method=='hdm_ds_single') {
    fits=lapply(ts,function(j) hdm::rlassoEffect(X[,-j,drop=FALSE],y,X[,j,drop=FALSE],method='double selection',post=TRUE));beta=vapply(fits,function(f) unname(f$alpha),numeric(1));se=vapply(fits,function(f) unname(f$se),numeric(1))
   } else {
    # Match the authors' initialization convention:10fold CV,standardizeFALSE.
    fit=cv.glmnet(X,y,family='binomial',standardize=FALSE,intercept=TRUE,nfolds=10)
    lev=if(method=='xia_refined') 'lambda.min' else 'lambda.1se';b=as.vector(coef(fit,s=lev));f=REF_DS_inf(X,y,'binomial',b);beta=f$est[ts+1];se=f$se[ts+1]
   }
   if(any(grepl('did not converge|not converge|numerically 0 or 1',warnings,ignore.case=TRUE))) stop('source convergence/separation warning')
   if(length(beta)!=length(ts)||any(!is.finite(beta))||any(!is.finite(se))||any(se<=0)) stop('invalid inference output')
   tab=data.frame(method=method,target=ts,truth=truth[ts],estimate=beta,se=se,lower=beta-qnorm(.975)*se,upper=beta+qnorm(.975)*se,pvalue=2*pnorm(-abs(beta/se)),converged=TRUE,seconds=proc.time()[[3]]-started,refit_status='native_single_API',initial_size=NA,lambda=NA,projection_parameter=NA)
   write.table(tab,out,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE)
  },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),paste0(stem,'.',method,'.error')))
  if(length(warnings))writeLines(warnings,paste0(stem,'.',method,'.warnings'))
 }
 writeLines('ok',paste0(stem,'.supp.complete'))
}
