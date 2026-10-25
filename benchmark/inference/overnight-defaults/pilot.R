args=commandArgs(TRUE);caseid=args[1];shard=as.integer(args[2]);maxrep=as.integer(args[3]);stage=args[4]
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/Rlib','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(hdm);library(glmnet);library(SIHR)})
cases=read.csv('cases.csv');family=cases$family[match(caseid,cases$case)];root=file.path('outputs',stage,caseid)
# A fully solved, target-specific double-selection candidate, with CV in both selections.
cvds=function(X,y,ts,oneSE=FALSE) {
 n=nrow(X);p=ncol(X);fam=if(family=='gaussian') 'gaussian' else 'binomial'
 folds=sample(rep(1:5,length.out=n));first=cv.glmnet(X,y,family=fam,foldid=folds,thresh=1e-9,maxit=100000)
 lev=if(oneSE) 'lambda.1se' else 'lambda.min';co=as.vector(coef(first,s=lev));S=which(co[-1]!=0)
 # Post-selection outcome fit estimates the Fisher weights for logistic auxiliary selection.
 init=glm.fit(cbind(1,X[,S,drop=FALSE]),y,family=if(family=='gaussian') gaussian() else binomial())
 if(!init$converged || any(!is.finite(init$coefficients))) stop('initial refit invalid')
 h=if(family=='gaussian') rep(1,n) else pmax(init$fitted.values*(1-init$fitted.values),1e-8)
 ans=matrix(NA_real_,length(ts),2)
 for(a in seq_along(ts)) {
  j=ts[a];N=X[,-j,drop=FALSE];d=X[,j]
  aux=cv.glmnet(N,d,weights=h,foldid=folds,thresh=1e-9,maxit=100000)
  Sd=which(as.vector(coef(aux,s=lev))[-1]!=0);others=setdiff(seq_len(p),j)
  union=union(setdiff(S,j),others[Sd]);if(length(union)+2>=n-5) stop('selected union too large')
  M=cbind(1,d,X[,union,drop=FALSE]);if(qr(M)$rank<ncol(M)) stop('rank deficient selected union')
  fit=glm.fit(M,y,family=if(family=='gaussian') gaussian() else binomial())
  if(!fit$converged || any(!is.finite(fit$coefficients)) || (family!='gaussian' && max(abs(fit$linear.predictors))>25)) stop('final refit invalid or separated')
  w=if(family=='gaussian') rep(1,n) else fit$fitted.values*(1-fit$fitted.values)
  H=crossprod(M,M*w);Hinv=solve(H);r=y-fit$fitted.values
  V=Hinv%*%crossprod(M,M*as.vector(r^2))%*%Hinv
  ans[a,]=c(fit$coefficients[2],sqrt(V[2,2]*n/(n-ncol(M))))
 }
 list(beta=ans[,1],se=ans[,2])
}
for(rep in seq(shard,maxrep,by=8)) {
 stem=file.path(root,rep);if(file.exists(paste0(stem,'.rcomplete'))) next
 if(!file.exists(paste0(stem,'.jcomplete'))) next
 X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));y=scan(file.path(stem,'y.csv'),quiet=TRUE);truth=scan(file.path(stem,'beta.csv'),quiet=TRUE);ts=scan(file.path(stem,'targets.csv'),quiet=TRUE)
 out=paste0(stem,'.R.csv');writeLines('method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter',out)
 methods=c('hdm_post','hdm_no_post',if(family=='gaussian') 'hdm_ds' else character(),'cvds_min','cvds_1se','SIHR',if(family!='gaussian') 'SIHR_alter' else character(),if(nrow(X)>ncol(X)+5) 'full_mle' else character())
 for(method in methods) {
  warnings=character();started=proc.time()[[3]]
  result=tryCatch(withCallingHandlers({
   set.seed(980000+rep)
   if(grepl('^hdm',method)) {
    if(family=='gaussian') {f=hdm::rlassoEffects(X,y,index=ts,method=if(method=='hdm_ds') 'double selection' else 'partialling out',post=method!='hdm_no_post');beta=as.vector(f$coefficients);se=as.vector(f$se)}
    else {f=hdm::rlassologitEffects(X,y,index=ts,post=method!='hdm_no_post');beta=as.vector(f$coefficients);se=as.vector(f$se)}
   } else if(grepl('^cvds',method)) {f=cvds(X,y,ts,method=='cvds_1se');beta=f$beta;se=f$se
   } else if(grepl('^SIHR',method)) {
    L=matrix(0,ncol(X),length(ts));L[cbind(ts,seq_along(ts))]=1
    f=SIHR::LF(X,y,L,model=if(family=='gaussian') 'linear' else if(method=='SIHR_alter') 'logistic_alter' else 'logistic');beta=f$est.debias.vec;se=f$se.vec
   } else {
    M=cbind(1,X);f=glm.fit(M,y,family=if(family=='gaussian') gaussian() else binomial());if(!f$converged) stop('MLE not converged')
    w=if(family=='gaussian') rep(1,nrow(X)) else f$fitted.values*(1-f$fitted.values);H=solve(crossprod(M,M*w));beta=f$coefficients[ts+1];se=sqrt(diag(H)[ts+1]*(if(family=='gaussian') sum(f$residuals^2)/(nrow(X)-ncol(M)) else 1))
   }
   if(any(grepl('did not converge|not converge|numerically 0 or 1',warnings,ignore.case=TRUE))) stop('source convergence/separation warning: see warning receipt')
   if(length(beta)!=length(ts)||any(!is.finite(beta))||any(!is.finite(se))||any(se<=0)) stop('nonfinite or missing results')
   tab=data.frame(method=method,target=ts,truth=truth[ts],estimate=beta,se=se,lower=beta-qnorm(.975)*se,upper=beta+qnorm(.975)*se,pvalue=2*pnorm(-abs(beta/se)),converged=TRUE,seconds=proc.time()[[3]]-started,refit_status='native',initial_size=NA,lambda=NA,projection_parameter=NA)
   write.table(tab,out,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE);TRUE
  },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){writeLines(conditionMessage(e),paste0(stem,'.',method,'.error'));FALSE})
  if(length(warnings))writeLines(warnings,paste0(stem,'.',method,'.warnings'))
 }
 writeLines('ok',paste0(stem,'.rcomplete'))
}
