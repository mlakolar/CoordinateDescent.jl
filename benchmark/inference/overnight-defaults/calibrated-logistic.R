# Experimental Appendix-A4 adaptation of Liu/Zhang/Zhou (2021).
# Native glmnet + RCAL solvers; Wald sandwich, not the paper's bootstrap CI.
args=commandArgs(TRUE);caseid=args[1];shard=as.integer(args[2]);maxrep=as.integer(args[3]);stage=args[4]
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/Rlib-calibration','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(glmnet);library(RCAL)})
cases=read.csv('../calibration-v1/calibration-cases.csv');case=cases[match(caseid,cases$case),]
stable_exp=function(x){if(any(!is.finite(x))||any(abs(x)>600))stop('exponential range invalid');exp(x)}
kkt=function(gradient,b,lambda) {
 nz=abs(b)>1e-9
 max(c(abs(gradient[1]),abs(gradient[-1][nz]+lambda*sign(b[nz])),pmax(abs(gradient[-1][!nz])-lambda,0)),na.rm=TRUE)
}
root_near=function(fn,start) {
 grid=sort(unique(c(seq(start-5,start+5,length.out=101),start)))
 values=vapply(grid,fn,0.);brackets=which(values[-length(values)]*values[-1]<=0)
 if(!length(brackets))stop('no finite bracket for calibrated score')
 k=brackets[which.min(abs((grid[brackets]+grid[brackets+1])/2-start))]
 ans=uniroot(fn,c(grid[k],grid[k+1]),tol=1e-10)$root
 if(!is.finite(ans)||abs(fn(ans))>1e-7)stop('calibrated root residual invalid')
 ans
}
cal_cv=function(y,N,w,offset,levels,foldid) {
 # RCAL normalizes weights to mean one; penalties here use its native units.
 n=nrow(N);p=ncol(N);W=w/mean(w);K=max(foldid)
 errors=matrix(NA_real_,length(levels),K);fits=vector('list',length(levels))
 for(k in seq_len(K)) {
  test=which(foldid==k);train=which(foldid!=k);initial=NULL
  for(a in seq_along(levels)) {
   f=tryCatch(RCAL::glm.regu(y[train],N[train,,drop=FALSE],iw=W[train],loss='cal',init=initial,rhos=rep(levels[a],p),offs=offset[train],n.iter=500,eps=1e-10,bt.lim=20),error=function(e)NULL)
   if(is.null(f)||f$conv<=0||any(!is.finite(c(f$inter,f$bet))))next
   initial=c(f$inter,f$bet);eta=offset[test]+cbind(1,N[test,,drop=FALSE])%*%initial
   loss=ifelse(y[test]==1,stable_exp(-eta),eta)
   errors[a,k]=mean(W[test]*loss)
  }
 }
 finite=which(apply(is.finite(errors),1,all));if(!length(finite))stop('no fully converged calibration CV level')
 meanloss=rowMeans(errors);best=finite[which.min(meanloss[finite])];mc=sd(errors[best,])/sqrt(K)
 # Map the largest eligible penalty back to the original grid index.
 eligible=finite[meanloss[finite]<=meanloss[best]+mc];oneSE=eligible[which.max(levels[eligible])]
 for(a in unique(c(best,oneSE))) {
  f=RCAL::glm.regu(y,N,iw=W,loss='cal',rhos=rep(levels[a],p),offs=offset,n.iter=500,eps=1e-10,bt.lim=20)
  if(f$conv<=0)stop('full calibration regression not converged')
  eta=offset+cbind(1,N)%*%c(f$inter,f$bet)
  grad=colMeans(cbind(1,N)*as.vector(W*((1-y)-y*stable_exp(-eta))))
  err=kkt(grad,f$bet,levels[a]);if(err>2e-5)stop(paste('calibration KKT residual',err))
  f$kkt=err;fits[[a]]=f
 }
 list(fits=fits,best=best,oneSE=oneSE,invalid_levels=sum(!apply(is.finite(errors),1,all)))
}
for(rep in seq(shard,maxrep,by=8)) {
 stem=file.path('outputs',stage,caseid,rep);if(file.exists(paste0(stem,'.cal.complete')))next
 if(case$design=='liu') {
  # Gaussian class-conditional Appendix-A5(i) design with 200 controls.
  # Bayes intercept is -.155; target alpha=.5. n=200 is an added stress case.
  set.seed((if(stage=='canary')93000000 else 94000000)+match(caseid,cases$case)*10000+rep)
  n=case$n;p=case$p;P=diag(c(1.5,rep(1.2,p-1)));P[1,2:5]=P[2:5,1]=.2
  U=chol(solve(P));y=rbinom(n,1,.5);mu=c(.4,-.25,-.25,rep(0,p-3))
  X=matrix(rnorm(n*p),n,p)%*%U+outer(y,mu);truth=as.vector(P%*%mu);ts=c(1,(p-4):p)
  dir.create(stem,recursive=TRUE,showWarnings=FALSE)
  for(name in c('X','y','truth','ts'))write.table(get(name),file.path(stem,paste0('cal-',name,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
 } else {
  X=as.matrix(read.csv(file.path(stem,'X.csv'),header=FALSE));y=scan(file.path(stem,'y.csv'),quiet=TRUE);truth=scan(file.path(stem,'beta.csv'),quiet=TRUE);ts=scan(file.path(stem,'targets.csv'),quiet=TRUE)
 }
 n=nrow(X);p=ncol(X);out=paste0(stem,'.cal.csv');di=paste0(stem,'.cal.diagnostics.csv')
 writeLines('method,target,truth,estimate,se,lower,upper,pvalue,converged,seconds,refit_status,initial_size,lambda,projection_parameter',out)
 writeLines('method,target,alpha_kkt,gamma_kkt,root_residual,derivative_error,initial_lambda,alpha_lambda,gamma_lambda,alpha_size,gamma_size,invalid_cv_levels',di)
 for(j in ts) {
  warnings=character();started=proc.time()[[3]]
  tryCatch(withCallingHandlers({
   set.seed(992000+rep+j);foldid=sample(rep(1:5,length.out=n));levels=exp(seq(log(2*sqrt(log(p)/n)),log(.2*sqrt(log(p)/n)),length.out=8))
   first=cv.glmnet(X,y,family='binomial',lambda=levels,foldid=foldid,standardize=FALSE,thresh=1e-10,maxit=1000000)
   co=as.vector(coef(first,s='lambda.min'));bstart=co[j+1];N=X[,-j,drop=FALSE];M=cbind(1,N);d=X[,j]
   initialgrad=colMeans(cbind(1,X)*as.vector(plogis(cbind(1,X)%*%co)-y))
   if(kkt(initialgrad,co[-1],first$lambda.min)>2e-5)stop('initial logistic KKT residual invalid')
   rtilde=as.vector(M%*%co[-(j+1)]);psi=plogis(-rtilde);weight=plogis(rtilde)
   wa=(1-y)*weight;aux=cv.glmnet(N,d,weights=wa,lambda=levels,foldid=foldid,standardize=FALSE,thresh=1e-10,maxit=1000000)
   ac=as.vector(coef(aux,s='lambda.min'));m=as.vector(M%*%ac);v=d-m
   agrad=-colMeans(M*as.vector((wa/mean(wa))*v));ae=kkt(agrad,ac[-1],aux$lambda.min)
   if(ae>2e-5)stop(paste('auxiliary KKT residual',ae))
   preliminary=function(beta)mean(psi*(y*stable_exp(-beta*d)-(1-y)*stable_exp(rtilde))*v)
   bpre=root_near(preliminary,bstart)
   gamma=cal_cv(y,N,weight,bpre*d,levels,foldid)
   for(policy in c('calibrated_min','calibrated_1se')) {
    idx=if(policy=='calibrated_min')gamma$best else gamma$oneSE;g=gamma$fits[[idx]];rhat=as.vector(M%*%c(g$inter,g$bet))
    score=function(beta)weight*(y*stable_exp(-beta*d-rhat)-(1-y))*v
    beta=root_near(function(beta)mean(score(beta)),bpre)
    derivative=-mean(weight*y*d*stable_exp(-beta*d-rhat)*v)
    fd=(mean(score(beta+1e-5))-mean(score(beta-1e-5)))/2e-5;de=abs(fd-derivative)/max(1,abs(derivative))
    if(derivative>=-1e-8||de>1e-6)stop('calibrated score derivative invalid')
    influence=score(beta)/derivative;se=sd(influence)/sqrt(n)
    if(!is.finite(se)||se<=0)stop('invalid calibrated standard error')
    if(any(grepl('did not converge|not converge|numerically 0 or 1',warnings,ignore.case=TRUE)))stop('source convergence warning')
    tab=data.frame(method=policy,target=j,truth=truth[j],estimate=beta,se=se,lower=beta-qnorm(.975)*se,upper=beta+qnorm(.975)*se,pvalue=2*pnorm(-abs(beta/se)),converged=TRUE,seconds=proc.time()[[3]]-started,refit_status='A4_calibration_Wald_adaptation',initial_size=sum(g$bet!=0),lambda=first$lambda.min,projection_parameter=levels[idx])
    write.table(tab,out,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE)
    dd=data.frame(method=policy,target=j,alpha_kkt=ae,gamma_kkt=g$kkt,root_residual=abs(mean(score(beta))),derivative_error=de,initial_lambda=first$lambda.min,alpha_lambda=aux$lambda.min,gamma_lambda=levels[idx],alpha_size=sum(ac[-1]!=0),gamma_size=sum(g$bet!=0),invalid_cv_levels=gamma$invalid_levels)
    write.table(dd,di,sep=',',row.names=FALSE,col.names=FALSE,append=TRUE,quote=FALSE)
   }
  },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),paste0(stem,'.target',j,'.cal.error')))
  if(length(warnings))writeLines(warnings,paste0(stem,'.target',j,'.cal.warnings'))
 }
 writeLines('attempted all targets; inspect failures and complete-target counts',paste0(stem,'.cal.complete'))
}
