a=commandArgs(TRUE);case=a[1];seed=as.integer(a[2]);eps=as.numeric(a[3]);rep=as.integer(a[4])
library(conquer);d=readRDS(file.path('inputs',paste0(case,'-',seed,'.rds')));set.seed(seed+10000L*rep)
fit=function(X,y,ls,tau) {
 z=conquer.reg(X,y,lambda=ls,tau=tau,kernel='parabolic',h=.2,penalty='lasso',epsilon=eps,iteMax=30000)
 B=as.matrix(z$coeff)[,match(ls,z$lambda),drop=FALSE]
 r=matrix(y,nrow(X),length(ls))-cbind(1,X)%*%B;s=pmax(-1,pmin(1,r/.2))
 list(B=B,U=tau-.5+.75*s-.25*s^3)
}
invisible(fit(d$X[1:60,1:12],d$y[1:60],c(.2,.1),d$tau));gc()
t0=as.numeric(Sys.time());z=fit(d$X,d$y,d$ls,d$tau);elapsed=as.numeric(Sys.time())-t0
out=file.path('results',paste(case,seed,rep,eps,sep='-'))
writeBin(as.double(z$B),paste0(out,'-B.bin'),size=8,endian='little');writeBin(as.double(z$U),paste0(out,'-U.bin'),size=8,endian='little')
writeLines(as.character(elapsed),paste0(out,'-seconds.txt'));writeLines(capture.output(sessionInfo()),paste0(out,'-Rsession.txt'))
