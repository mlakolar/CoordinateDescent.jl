# Development variant: reject numerical recession witnesses and uncertified CV fits.
# Same loss, grid, folds and fitted nuisance inputs as the original engine study.
a=commandArgs(TRUE);caseid=a[1];rep=as.integer(a[2]);rs=a[2]
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/Rlib-calibration',.libPaths()))
suppressPackageStartupMessages(library(RCAL))
root=file.path('outputs','calibration-cv-engine',caseid,rs);out=file.path('outputs','calibration-cv-bounded',caseid,rs);dir.create(out,recursive=TRUE,showWarnings=FALSE)
readv=function(name)scan(file.path(root,paste0(name,'.csv')),quiet=TRUE,sep=',')
base=file.path('outputs','development',caseid,rs);liu=grepl('^liu-',caseid)
X=as.matrix(read.csv(file.path(base,if(liu)'cal-X.csv' else 'X.csv'),header=FALSE));y=scan(file.path(base,if(liu)'cal-y.csv' else 'y.csv'),quiet=TRUE)
j=as.integer(readv('j'));N=X[,-j,drop=FALSE];W=readv('weight');W=W/mean(W);offset=readv('offset');levels=readv('levels');foldid=as.integer(readv('foldid'));K=max(foldid)
checks=read.csv(file.path('outputs','calibration-recession',caseid,rs,'mask.csv'))
bad=function(a,k)any(checks$scope=='actual_cv'&checks$fold==k&checks$level==a&checks$unbounded_witness)
kkt=function(grad,b,lambda){nz=abs(b)>1e-9;max(c(abs(grad[1]),abs(grad[-1][nz]+lambda*sign(b[nz])),pmax(abs(grad[-1][!nz])-lambda,0)))}
errors=matrix(NA_real_,length(levels),K);diagnostics=data.frame();warn=character();started=proc.time()[[3]]
tryCatch(withCallingHandlers({
 for(k in seq_len(K)) {
  test=which(foldid==k);train=which(foldid!=k);init=NULL;Wt=W[train]/mean(W[train]);M=cbind(1,N[train,,drop=FALSE])
  for(a in seq_along(levels)) {
   if(bad(a,k))next
   f=tryCatch(RCAL::glm.regu(y[train],N[train,,drop=FALSE],iw=W[train],loss='cal',init=init,rhos=rep(levels[a],ncol(N)),offs=offset[train],n.iter=500,eps=1e-10,bt.lim=20),error=function(e)NULL)
   if(is.null(f))next
   b=c(f$inter,f$bet);eta=as.vector(offset[train]+M%*%b)
   grad=colMeans(M*as.vector(Wt*((1-y[train])-y[train]*exp(-eta))));err=kkt(grad,f$bet,levels[a])
   diagnostics=rbind(diagnostics,data.frame(fold=k,level=a,converged=f$conv,kkt=err))
   if(f$conv<=0||!is.finite(err)||err>2e-5)next
   init=b;eta=as.vector(offset[test]+cbind(1,N[test,,drop=FALSE])%*%b)
   value=mean(W[test]*(y[test]*exp(-eta)+(1-y[test])*eta));if(is.finite(value))errors[a,k]=value
  }
 }
 write.table(errors,file.path(out,'rcal-errors.csv'),sep=',',row.names=FALSE,col.names=FALSE)
 write.csv(diagnostics,file.path(out,'rcal-fold-checks.csv'),row.names=FALSE)
 finite=which(apply(is.finite(errors),1,all));finite=finite[!vapply(finite,bad,FALSE,k=0)]
 if(!length(finite))stop('no fully certified bounded native CV level')
 means=rowMeans(errors);best=finite[which.min(means[finite])];mc=sd(errors[best,])/sqrt(K);eligible=finite[means[finite]<=means[best]+mc];oneSE=eligible[which.max(levels[eligible])]
 for(policy in c('best','oneSE')) {
  idx=get(policy);f=RCAL::glm.regu(y,N,iw=W,loss='cal',rhos=rep(levels[idx],ncol(N)),offs=offset,n.iter=500,eps=1e-10,bt.lim=20)
  b=c(f$inter,f$bet);M=cbind(1,N);eta=as.vector(offset+M%*%b);grad=colMeans(M*as.vector(W*((1-y)-y*exp(-eta))));err=kkt(grad,f$bet,levels[idx])
  if(f$conv<=0||!is.finite(err)||err>2e-5)stop('native full fit uncertified')
  obj=mean(W*(y*exp(-eta)+(1-y)*eta))+levels[idx]*sum(abs(f$bet))
  write.table(b,file.path(out,paste0('rcal-',policy,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
  write.csv(data.frame(index=idx,lambda=levels[idx],kkt=err,objective=obj,seconds=proc.time()[[3]]-started),file.path(out,paste0('rcal-',policy,'.meta.csv')),row.names=FALSE)
 }
},warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),file.path(out,'rcal.error')))
if(length(warn))writeLines(warn,file.path(out,'rcal.warnings'))
