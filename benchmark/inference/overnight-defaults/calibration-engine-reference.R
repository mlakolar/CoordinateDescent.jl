# Fixed-penalty native RCAL reference on saved development datasets, not CV or inference.
a=commandArgs(TRUE);caseid=a[1];rep=as.integer(a[2])
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/Rlib-calibration',.libPaths()))
suppressPackageStartupMessages(library(RCAL))
stem=file.path('outputs','development',caseid,rep)
liu=grepl('^liu-',caseid)
getdata=function(name)as.matrix(read.csv(file.path(stem,paste0(if(liu)'cal-' else '',name,'.csv')),header=FALSE))
X=getdata('X');y=as.vector(getdata('y'));n=nrow(X);p=ncol(X)
d=X[,1];N=X[,-1,drop=FALSE];M=cbind(1,N)
W=plogis(.3*N[,1]-.2*N[,2]);W=W/mean(W);offs=.2*d
root=file.path('outputs','calibration-engine',caseid,rep);dir.create(root,recursive=TRUE,showWarnings=FALSE)
write.table(W,file.path(root,'weights.csv'),sep=',',row.names=FALSE,col.names=FALSE)
write.table(offs,file.path(root,'offset.csv'),sep=',',row.names=FALSE,col.names=FALSE)
warn=character()
for(mult in c(2,.7)) {
 tag=as.character(mult);lambda=mult*sqrt(log(p)/n);started=proc.time()[[3]]
 tryCatch(withCallingHandlers({
 f=RCAL::glm.regu(y,N,iw=W,loss='cal',rhos=rep(lambda,p-1),offs=offs,n.iter=500,eps=1e-10,bt.lim=20)
 elapsed=proc.time()[[3]]-started;b=c(f$inter,f$bet);eta=as.vector(offs+M%*%b)
 grad=colMeans(M*as.vector(W*((1-y)-y*exp(-eta))));nz=abs(b[-1])>1e-9
 kkt=max(c(abs(grad[1]),abs(grad[-1][nz]+lambda*sign(b[-1][nz])),pmax(abs(grad[-1][!nz])-lambda,0)))
 obj=mean(W*(y*exp(-eta)+(1-y)*eta))+lambda*sum(abs(b[-1]))
 write.table(b,file.path(root,paste0('rcal-',tag,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
 meta=data.frame(lambda=lambda,converged=f$conv,kkt=kkt,objective=obj,seconds=elapsed)
 write.csv(meta,file.path(root,paste0('rcal-',tag,'.meta.csv')),row.names=FALSE)
 if(f$conv<=0||!is.finite(kkt)||kkt>2e-5)stop('RCAL reference not certified')
 },warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),file.path(root,paste0('rcal-',tag,'.error'))))
}
if(length(warn))writeLines(warn,file.path(root,'rcal.warnings'))
