.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages(library(glmnet))
X=as.matrix(read.csv('source-v3/test/data/inference/X.csv'))
X=sweep(X,2,colMeans(X));scales=sqrt(colMeans(X^2));A=sweep(X,2,scales,'/')
n=nrow(A);p=ncol(A);levels=c(.4,.3,.2,.15,.1,.07,.05,.03,.02,.01,.005,.001)
for(m in c(0,4)){
 M=matrix(0,p,p);diagnostic=matrix(0,p,5)
 for(j in seq_len(p)){
  others=setdiff(seq_len(p),j);selected=if(m)others[order(-abs(crossprod(A[,others,drop=FALSE],A[,j])),others)[seq_len(m)]] else integer(0)
  rest=setdiff(others,selected)
  B=A[,selected,drop=FALSE];C=A[,rest,drop=FALSE];yy=A[,j]
  if(m){C=C-B%*%qr.solve(B,C);yy=yy-B%*%qr.solve(B,yy)}
  g=glmnet(C,yy,lambda=levels,intercept=FALSE,standardize=FALSE,control=list(thresh=1e-14,maxit=100000))
  gam=as.matrix(coef(g))[-1,,drop=FALSE]
  V=matrix(0,p,length(levels));V[j,]=1;V[rest,]=-gam
  if(m)V[selected,]=-qr.solve(B,A[,j]-A[,rest,drop=FALSE]%*%gam)
  R=A%*%V;lengths=sqrt(colSums(R^2));eta=apply(abs(crossprod(A[,others,drop=FALSE],R)),2,max)/lengths
  tau=lengths/abs(colSums(A[,j]*R));target=sqrt(2*log(p))
  feasible=which(eta<=target);if(!length(feasible))feasible=which(eta<=1.1*min(eta))
  ref=feasible[1];chosen=max(which(seq_along(levels)>=ref & tau<=1.25*tau[ref]))
  den=sum(A[,j]*R[,chosen])/n;M[j,]=V[,chosen]/(den*scales*scales[j])
  diagnostic[j,]=c(levels[chosen],eta[chosen],tau[chosen]/scales[j],tau[ref]/scales[j],eta[chosen]<=target)
 }
 write.table(M,paste0('restricted-reference-theta-',m,'.csv'),sep=',',row.names=FALSE,col.names=FALSE)
 write.table(diagnostic,paste0('restricted-reference-diagnostics-',m,'.csv'),sep=',',row.names=FALSE,col.names=FALSE)
}
