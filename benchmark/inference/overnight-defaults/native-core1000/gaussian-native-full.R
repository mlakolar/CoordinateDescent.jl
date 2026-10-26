# Native out-of-box R recipes on exactly regenerated Gaussian campaign datasets.
a=commandArgs(TRUE);engine=a[1];caseid=a[2];rep=as.integer(a[3])
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
base=file.path('competitor-data',caseid,rep);out=file.path('outputs','gaussian-native-full',caseid,paste0(rep,'.',engine));dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
X=as.matrix(read.csv(file.path(base,'X.csv'),header=FALSE));y=scan(file.path(base,'y.csv'),quiet=TRUE);ts=scan(file.path(base,'targets.csv'),quiet=TRUE);truth=scan(file.path(base,'truth.csv'),quiet=TRUE)
suppressPackageStartupMessages(library(glmnet))
if(engine=='hdi')suppressPackageStartupMessages(library(hdi))
direction.calls=list()
if(engine=='SIHR') {
 suppressPackageStartupMessages(library(SIHR));ns=asNamespace('SIHR');original.direction=get('Direction_fixedtuning',ns)
 tracked.direction=function(...) {ans=original.direction(...);direction.calls[[length(direction.calls)+1L]]<<-list(mu=ans$mu,status=ans$status);ans}
 unlockBinding('Direction_fixedtuning',ns);assign('Direction_fixedtuning',tracked.direction,ns);lockBinding('Direction_fixedtuning',ns)
}
if(engine=='hdm')suppressPackageStartupMessages(library(hdm))
author=new.env();author$last_M=NULL;author$last_G=NULL
if(engine=='sslasso') {
 suppressPackageStartupMessages({library(Matrix);library(expm);library(flare)})
 src=readLines('/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/references/sslasso_code/lasso_inference.r')
 # Observation-only instrumentation: preserve all numerical statements and defaults,
 # including known diagonal and explicit-lambda bugs. Native lambda is omitted.
 src=sub('M[i,] <- beta;','selected_mu[i] <<- mu; M[i,] <- beta;',src,fixed=TRUE)
 src=sub('unbiased.Lasso <- as.numeric','last_M <<- M; last_G <<- sigma.hat; unbiased.Lasso <- as.numeric',src,fixed=TRUE)
 eval(parse(text=src[!grepl('^library',src)]),envir=author)
}
fit=function(A,b,targets) {
 p=ncol(A)
 if(engine=='hdi') {f=hdi::lasso.proj(A,b,family='gaussian');return(list(beta=f$bhat[targets],se=f$se[targets],lower=f$bhat[targets]-qnorm(.975)*f$se[targets],upper=f$bhat[targets]+qnorm(.975)*f$se[targets],pv=f$pval[targets],actual=p))}
 if(engine=='SIHR') {L=matrix(0,p,length(targets));L[cbind(targets,seq_along(targets))]=1;f=SIHR::LF(A,b,loading.mat=L,model='linear');return(list(beta=f$est.debias.vec,se=f$se.vec,lower=f$est.debias.vec-qnorm(.975)*f$se.vec,upper=f$est.debias.vec+qnorm(.975)*f$se.vec,pv=2*pnorm(-abs(f$est.debias.vec/f$se.vec)),actual=length(targets)))}
 if(engine=='hdm') {
  # Documented single-target public API; retain earlier multi-wrapper failure separately.
  ff=lapply(targets,function(j)hdm::rlassoEffect(A[,-j,drop=FALSE],b,A[,j],method='double selection'))
  beta=vapply(ff,function(f)as.numeric(f$coefficients)[1],0.);se=vapply(ff,function(f)as.numeric(f$se)[1],0.)
  return(list(beta=beta,se=se,lower=beta-qnorm(.975)*se,upper=beta+qnorm(.975)*se,pv=2*pnorm(-abs(beta/se)),actual=length(targets)))
 }
 if(engine=='sslasso') {
  author$selected_mu=rep(NA_real_,p+1)
  f=author$SSLasso(A,b,verbose=FALSE)
  # Returned vector follows the author's public API, which drops its intercept.
  return(list(beta=f$unb.coef[targets],se=rep(NA_real_,length(targets)),lower=f$low.lim[targets],upper=f$up.lim[targets],pv=f$pvals[targets],actual=p))
 }
 stop('unknown method')
}
warnings=character();status=list(status='error');set.seed(177)
tryCatch(withCallingHandlers({
 A=matrix(rnorm(80*12),80,12);A=scale(A,scale=FALSE);fit(A,rnorm(80),c(1,2));warnings=character();direction.calls=list();set.seed(770000+rep);gc()
 t=proc.time()[[3]];f=fit(X,y,ts);elapsed=proc.time()[[3]]-t
 ans=data.frame(method=engine,target=ts,truth=truth[ts],estimate=f$beta,se=f$se,lower=f$lower,upper=f$upper,pvalue=f$pv,converged=TRUE,seconds=elapsed,refit_status='native_default',initial_size=NA,lambda=NA,projection_parameter=NA)
 if(!all(is.finite(as.matrix(ans[,c('estimate','lower','upper','pvalue')]))))stop('nonfinite native output')
 write.csv(ans,paste0(out,'.csv'),row.names=FALSE)
 status=list(status='ok',fit_seconds=elapsed,actual_inferred_targets=f$actual)
 if(engine=='SIHR' && length(direction.calls)) {status$projection_statuses=paste(unique(vapply(direction.calls,function(x)as.character(x$status),'')),collapse=' | ');status$projection_mu_min=min(vapply(direction.calls,function(x)x$mu,0.));status$projection_mu_max=max(vapply(direction.calls,function(x)x$mu,0.))}
 if(engine=='sslasso') {
  M=author$last_M;G=author$last_G;mu=author$selected_mu
  stopifnot(is.matrix(M),is.matrix(G),nrow(M)==ncol(X)+1)
  if(rep<=20)write.table(M,paste0(out,".theta.csv"),sep=",",row.names=FALSE,col.names=FALSE)
  if(all(is.na(mu)))mu=rep(0,nrow(M))
  residual=apply(abs(M%*%G-diag(ncol(G))),1,max)
  gap=rowSums((M%*%G)*M)-diag(M)+mu*rowSums(abs(M))
  certificate=data.frame(row=seq_len(nrow(M)),mu=mu,residual=residual,gap=gap,certified=is.finite(mu)&residual<=mu+1e-6&abs(gap)<1e-6)
  write.csv(certificate,paste0(out,'.qp.csv'),row.names=FALSE)
  status$all_qp_rows_certified=all(certificate$certified);status$qp_max_abs_gap=max(abs(gap))
 }
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){status<<-list(status='error',error=conditionMessage(e))})
status$warnings=paste(unique(warnings),collapse=' | ')
writeLines(paste(names(status),unlist(status),sep='='),paste0(out,'.meta'))
writeLines('attempt complete; inspect native status, warnings and certificates',paste0(out,'.complete'))
