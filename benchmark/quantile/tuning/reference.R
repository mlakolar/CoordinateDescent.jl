library(quantreg)
dir.create('reference',showWarnings=FALSE)
set.seed(419)
n=83L;p=5L;X=sweep(matrix(rnorm(n*p),n,p),2,1:p,'*');X=sweep(X,2,1:p,'+')
y=as.numeric(.8+X%*%c(1,-.5,0,0,0)+rnorm(n));o=.2*sin(1:n);fl=(1:n)%%3+1L
U=matrix(runif(n*128),n);w=c(.7,1,1.2,.8,2);ls=c(.3,.15,.05)
wb=function(x,name)writeBin(as.double(x),file.path('reference',paste0(name,'.bin')),size=8,endian='little')
wb(X,'X');wb(y,'y');wb(o,'offset');wb(U,'uniforms')
results=list()
fit.dual=rq.fit.lasso
body(fit.dual)=substitute({ans=ORIGINAL;ans$dual=z$wn[2L*N+seq_len(n)]-(1-tau);ans},list(ORIGINAL=body(fit.dual)))
taus=if('regular' %in% commandArgs(TRUE)) c(.23,.51,.79) else c(.2,.5,.8)
for(tau in taus) for(std in c(FALSE,TRUE)) {
 key=paste(tau,std,sep='-')
 fit=function(idx,dual=FALSE) {
  A=X[idx,,drop=FALSE];effective=w
  if(std)effective=effective*sqrt(colMeans(sweep(A,2,colMeans(A),'-')^2))
  ans=lapply(ls,function(lambda)fit.dual(cbind(1,A),y[idx]-o[idx],tau=tau,lambda=c(0,2*length(idx)*lambda*effective),eps=1e-9))
  B=vapply(ans,function(z)as.numeric(z$coefficients),numeric(p+1))
  if(dual)list(B=B,U=vapply(ans,function(z)as.numeric(z$dual),numeric(length(idx)))) else B
 }
 B=fit(1:n);E=matrix(NA_real_,3,length(ls))
 for(k in 1:3) {
  tr=which(fl!=k);te=which(fl==k);ans=fit(tr,TRUE);BF=ans$B
  wb(BF,paste0(key,'-fold',k,'-coef'));wb(ans$U,paste0(key,'-fold',k,'-dual'))
  R=matrix(y[te]-o[te],length(te),length(ls))-cbind(1,X[te,,drop=FALSE])%*%BF
  E[k,]=colMeans(pmax(tau*R,(tau-1)*R))
 }
 wb(B,paste0(key,'-coef'));wb(E,paste0(key,'-fold-errors'))
 effective=if(std)w*sqrt(colMeans(sweep(X,2,colMeans(X),'-')^2)) else w
 Z=sweep(X,2,colMeans(X),'-');S=tau-(U<tau)
 maxima=apply(abs(t(Z)%*%S)/(n*effective),2,max)
 results[[key]]=data.frame(tau=tau,standardize=std,lambda=2*quantile(maxima,.9,type=7,names=FALSE))
}
write.csv(do.call(rbind,results),'reference/calibration.csv',row.names=FALSE)
writeLines(capture.output(sessionInfo()),'reference/Rsession.txt')
cat('Independent R references complete\n')
