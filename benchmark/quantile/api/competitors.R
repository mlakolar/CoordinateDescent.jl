args=commandArgs(TRUE);mode=args[1];case=args[2];seed=as.integer(args[3]);base=file.path('inputs',paste(case,seed,sep='-'))
if(mode=='generate') {
 n=if(case=='tall') 10000L else 1000L
 p=switch(case,small=100L,tail=100L,tall=100L,wide=1000L,deepwide=3000L)
 tau=if(case=='tail') .1 else .5;levels=if(case=='deepwide') 16L else 32L
 set.seed(seed);X=matrix(rnorm(n*p),n,p);rho=if(case %in% c('tail','wide')) .8 else .5
 for(j in 2:p) X[,j]=rho*X[,j-1]+sqrt(1-rho^2)*X[,j]
 X=scale(X);e=if(case=='tail') rt(n,df=3)*(.5+.5*abs(X[,1])) else rnorm(n)
 y=as.numeric(.7+X[,1:10]%*%seq(1,.3,length.out=10)+e)
 a=sort(y)[ceiling(n*tau)];u=tau-(y<a);z=which(y==a);u[z]=u[z]-sum(u)/length(z)
 lm=max(abs(crossprod(X,u)))/n
 ls=exp(seq(log(lm*1.05),log(lm*if(case=='deepwide') .02 else .01),length.out=levels))
 dir.create('inputs',showWarnings=FALSE);dir.create('results',showWarnings=FALSE)
 saveRDS(list(X=X,y=y,ls=ls,tau=tau,n=n,p=p),paste0(base,'.rds'),compress=FALSE)
 for(z in c('X','y','ls')) writeBin(as.double(get(z)),paste0(base,'-',z,'.bin'),size=8,endian='little')
 writeLines(paste(n,p,tau,levels,sep=','),paste0(base,'-meta.csv'))
 methods=c('quantreg-native-dual','rqpen-br','rqpen-fn','rqpen-huber','conquer-dual')
 ord=do.call(rbind,lapply(1:2,function(rep) cbind(rep,sample(methods))))
 write.table(ord,paste0(base,'-order.tsv'),sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
 quit(status=0)
}
method=args[4];rep=as.integer(args[5]);d=readRDS(paste0(base,'.rds'))
library(quantreg);library(conquer);library(rqPen)
# rqfnb swaps the bounded LP scores x into wn column3 before return.
# Instrument a local copy only; do not modify the installed package namespace.
# Its original public wrapper discards this native workspace.
fn.dual=quantreg::rq.fit.lasso
body(fn.dual)=substitute({ans=ORIGINAL;ans$dual=z$wn[2L*N+seq_len(n)]-(1-tau);ans},list(ORIGINAL=body(fn.dual)))
fit=function(X,y,ls,tau) {
 n=nrow(X);p=ncol(X);U=NULL;gamma=NA_real_
 if(method=='quantreg-native-dual') {
  ans=lapply(ls,function(lambda) fn.dual(cbind(1,X),y,tau=tau,lambda=2*n*lambda,eps=1e-8))
  B=vapply(ans,function(z) as.numeric(z$coefficients),numeric(p+1));U=vapply(ans,function(z) z$dual,numeric(n))
 } else if(method=='conquer-dual') {
  ans=conquer.reg(X,y,lambda=ls,tau=tau,kernel='parabolic',h=.2,penalty='lasso',epsilon=1e-7,iteMax=30000)
  B=as.matrix(ans$coeff)[,match(ls,ans$lambda),drop=FALSE]
  R=matrix(y,n,length(ls))-cbind(1,X)%*%B;s=pmax(-1,pmin(1,R/.2));U=tau-.5+.75*s-.25*s^3
 } else {
  alg=switch(method,'rqpen-br'='br','rqpen-fn'='fn','rqpen-huber'='huber')
  ans=rq.pen(X,y,tau=tau,lambda=ls,penalty='LASSO',alg=alg,scalex=FALSE,lambda.discard=FALSE,coef.cutoff=0,converge.eps=1e-8,max.iter=30000)
  B=as.matrix(ans$models[[1]]$coefficients)
  if(!is.null(ans$lambda)) B=B[,match(ls,ans$lambda),drop=FALSE]
  if(ncol(B)!=length(ls)) stop('rqPen returned an incomplete lambda grid')
 }
 list(B=B,U=U,gamma=gamma)
}
invisible(fit(d$X[1:60,1:12],d$y[1:60],c(.2,.1),d$tau))
gc();setTimeLimit(elapsed=60,transient=TRUE);start=as.numeric(Sys.time());ans=fit(d$X,d$y,d$ls,d$tau);elapsed=as.numeric(Sys.time())-start
setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
out=file.path('results',paste(case,seed,rep,method,sep='-'))
writeBin(as.double(ans$B),paste0(out,'-B.bin'),size=8,endian='little')
if(!is.null(ans$U)) writeBin(as.double(ans$U),paste0(out,'-U.bin'),size=8,endian='little')
writeLines(as.character(elapsed),paste0(out,'-seconds.txt'))
writeLines(capture.output(sessionInfo()),paste0(out,'-Rsession.txt'))
cat('DONE',method,elapsed,'seconds\n')
