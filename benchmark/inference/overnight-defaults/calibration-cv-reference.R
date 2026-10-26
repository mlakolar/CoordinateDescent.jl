# Native RCAL calibration-CV audit at actual fitted target offsets/weights.
a=commandArgs(TRUE);caseid=a[1];rep=as.integer(a[2])
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261026-cd77-defaults/Rlib-calibration','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(glmnet);library(RCAL)})
src=readLines('../calibration-v2/calibrated-logistic-targets.R')
start=grep('^stable_exp=',src);end=grep('^for\\(rep in',src)-1
# Only expose the actual CV loss matrix for auditing, without changing fitting.
block=paste(src[start:end],collapse='\n')
block=sub('list(fits=fits,best=best,oneSE=oneSE,invalid_levels=', 'list(errors=errors,fits=fits,best=best,oneSE=oneSE,invalid_levels=',block,fixed=TRUE)
eval(parse(text=block))
stem=file.path('outputs','development',caseid,rep);liu=grepl('^liu-',caseid)
readmat=function(nm)as.matrix(read.csv(file.path(stem,paste0(if(liu)'cal-' else '',nm,'.csv')),header=FALSE))
X=readmat('X');y=as.vector(readmat('y'));n=nrow(X);p=ncol(X)
ts=as.vector(readmat(if(liu)'ts' else 'targets'));j=ts[1];N=X[,-j,drop=FALSE];M=cbind(1,N);d=X[,j]
root=file.path('outputs','calibration-cv-engine',caseid,rep);dir.create(root,recursive=TRUE,showWarnings=FALSE)
warn=character();started=proc.time()[[3]]
tryCatch(withCallingHandlers({
 set.seed(992000+rep+j);foldid=sample(rep(1:5,length.out=n));levels=exp(seq(log(2*sqrt(log(p)/n)),log(.2*sqrt(log(p)/n)),length.out=8))
 first=cv.glmnet(X,y,family='binomial',lambda=levels,foldid=foldid,standardize=FALSE,thresh=1e-10,maxit=1000000)
 co=as.vector(coef(first,s='lambda.min'));bstart=co[j+1];rtilde=as.vector(M%*%co[-(j+1)]);psi=plogis(-rtilde);weight=plogis(rtilde)
 wa=(1-y)*weight;aux=cv.glmnet(N,d,weights=wa,lambda=levels,foldid=foldid,standardize=FALSE,thresh=1e-10,maxit=1000000)
 ac=as.vector(coef(aux,s='lambda.min'));v=d-as.vector(M%*%ac)
 preliminary=function(beta)mean(psi*(y*stable_exp(-beta*d)-(1-y)*stable_exp(rtilde))*v)
 bpre=root_near(preliminary,bstart);offset=bpre*d
 # Preserve all inputs before the expensive native calibration CV.
 for(name in c('weight','offset','levels','foldid','j'))write.table(get(name),file.path(root,paste0(name,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
 gam=cal_cv(y,N,weight,offset,levels,foldid)
 write.table(gam$errors,file.path(root,'rcal-errors.csv'),sep=',',row.names=FALSE,col.names=FALSE)
 for(pol in c('best','oneSE')) {
  idx=gam[[pol]];g=gam$fits[[idx]];b=c(g$inter,g$bet);eta=offset+M%*%b;W=weight/mean(weight)
  obj=mean(W*(y*exp(-eta)+(1-y)*eta))+levels[idx]*sum(abs(g$bet))
  write.table(b,file.path(root,paste0('rcal-',pol,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
  write.csv(data.frame(index=idx,lambda=levels[idx],kkt=g$kkt,objective=obj,seconds=proc.time()[[3]]-started),file.path(root,paste0('rcal-',pol,'.meta.csv')),row.names=FALSE)
 }
},warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)writeLines(conditionMessage(e),file.path(root,'rcal.error')))
if(length(warn))writeLines(warn,file.path(root,'rcal.warnings'))
