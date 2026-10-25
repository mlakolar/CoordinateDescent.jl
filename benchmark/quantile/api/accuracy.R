args=commandArgs(TRUE);case=args[1];seed=as.integer(args[2]);epsilon=as.double(args[3]);rep=as.integer(args[4]);label=args[5]
library(conquer);library(quantreg)
fn.dual=quantreg::rq.fit.lasso
body(fn.dual)=substitute({ans=ORIGINAL;ans$dual=z$wn[2L*N+seq_len(n)]-(1-tau);ans},list(ORIGINAL=body(fn.dual)))
d=readRDS(file.path('inputs',paste0(case,'-',seed,'.rds')))
fit=function(X,y,ls,tau) {
 if(label=='quantreg-precomputed') {
  design=cbind(1,X);n=nrow(X);p=ncol(X)
  ans=lapply(ls,function(lambda) fn.dual(design,y,tau=tau,lambda=2*n*lambda,eps=1e-8))
  return(list(B=vapply(ans,function(z) as.numeric(z$coefficients),numeric(p+1)),U=vapply(ans,function(z) z$dual,numeric(n))))
 }
 a=conquer.reg(X,y,lambda=ls,tau=tau,kernel='parabolic',h=.2,penalty='lasso',epsilon=epsilon,iteMax=30000)
 B=as.matrix(a$coeff)[,match(ls,a$lambda),drop=FALSE]
 R=matrix(y,nrow(X),length(ls))-cbind(1,X)%*%B;s=pmax(-1,pmin(1,R/.2))
 list(B=B,U=tau-.5+.75*s-.25*s^3)
}
set.seed(seed+10000L*rep)
invisible(fit(d$X[1:60,1:12],d$y[1:60],c(.2,.1),d$tau))
gc();start=as.numeric(Sys.time());a=fit(d$X,d$y,d$ls,d$tau);elapsed=as.numeric(Sys.time())-start
base=file.path('accuracy',paste(case,seed,rep,label,sep='-'))
writeBin(as.double(a$B),paste0(base,'-B.bin'),size=8,endian='little')
writeBin(as.double(a$U),paste0(base,'-U.bin'),size=8,endian='little')
writeLines(as.character(elapsed),paste0(base,'-seconds.txt'))
