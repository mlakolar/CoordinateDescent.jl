.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages(library(glmnet))
X=as.matrix(read.csv('source-v1/test/data/inference/X.csv'))
X=sweep(X,2,colMeans(X));scales=sqrt(colMeans(X^2));A=sweep(X,2,scales,'/')
n=nrow(A);p=ncol(A);levels=c(.4,.3,.2,.15,.1,.07,.05,.03,.02,.01,.005,.001)
M=matrix(0,p,p);diagnostic=matrix(0,p,5)
for(j in seq_len(p)){
 g=glmnet(A[,-j,drop=FALSE],A[,j],lambda=levels,intercept=FALSE,standardize=FALSE,thresh=1e-14,maxit=100000)
 R=A[,j]-as.matrix(predict(g,A[,-j,drop=FALSE]))
 lengths=sqrt(colSums(R^2));eta=apply(abs(crossprod(A[,-j,drop=FALSE],R)),2,max)/lengths
 tau=lengths/abs(colSums(A[,j]*R));target=sqrt(2*log(p))
 feasible=which(eta<=target)
 if(!length(feasible))feasible=which(eta<=1.1*min(eta))
 ref=feasible[1];chosen=max(which(seq_along(levels)>=ref & tau<=1.25*tau[ref]))
 beta=as.vector(as.matrix(coef(g))[-1,chosen])
 v=rep(0,p);v[j]=1;v[-j]=-beta
 den=sum(A[,j]*R[,chosen])/n
 M[j,]=v/(den*scales*scales[j])
 diagnostic[j,]=c(levels[chosen],eta[chosen],tau[chosen]/scales[j],tau[ref]/scales[j],eta[chosen]<=target)
}
write.table(M,'ldpe-reference-theta.csv',sep=',',row.names=FALSE,col.names=FALSE)
write.table(diagnostic,'ldpe-reference-diagnostics.csv',sep=',',row.names=FALSE,col.names=FALSE)
