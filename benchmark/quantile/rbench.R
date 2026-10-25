args=commandArgs(TRUE);case=args[1];seed=as.integer(args[2]);out=args[3]
library(quantreg);library(conquer)
n=switch(case,small=1000L,tall=10000L,tall100k=100000L);p=100L
set.seed(seed);X=matrix(rnorm(n*p),n,p)
for(j in 2:p) X[,j]=.5*X[,j-1]+sqrt(.75)*X[,j]
X=scale(X);y=as.numeric(.7+X[,1:10]%*%seq(1,.3,length.out=10)+rnorm(n))
lm=max(abs(crossprod(X,.5-(y<median(y)))))/n
ls=exp(seq(log(lm*1.05),log(lm*.1),length.out=8))
base=file.path(out,paste(case,seed,sep='-'));dir.create(out,showWarnings=FALSE,recursive=TRUE)
writeBin(as.double(X),paste0(base,'-X.bin'),size=8,endian='little')
writeBin(y,paste0(base,'-y.bin'),size=8,endian='little')
writeBin(ls,paste0(base,'-lambda.bin'),size=8,endian='little')
# Load and compile before timing; the lasso penalty of quantreg is lambda/2.
invisible(rq.fit.lasso(cbind(1,X[1:50,1:12]),y[1:50],lambda=2*50*.1))
invisible(conquer.reg(X[1:50,1:12],y[1:50],lambda=.1,kernel='parabolic',h=.2))
rows=list()
for(rep in 1:2) for(method in sample(c('quantreg-fn','conquer'),2)) {
 gc();start=as.numeric(Sys.time())
 if(method=='quantreg-fn') {
  b=vapply(ls,function(lambda) as.numeric(rq.fit.lasso(cbind(1,X),y,tau=.5,lambda=2*n*lambda,eps=1e-7)$coefficients),numeric(p+1))
 } else {
  fit=conquer.reg(X,y,lambda=ls,tau=.5,kernel='parabolic',h=.2,penalty='lasso',epsilon=1e-6,iteMax=20000)
  b=as.matrix(fit$coeff)[,match(ls,fit$lambda),drop=FALSE]
 }
 elapsed=as.numeric(Sys.time())-start
 writeBin(as.double(b),paste0(base,'-',method,'-',rep,'.bin'),size=8,endian='little')
 rows[[length(rows)+1]]=data.frame(case=case,seed=seed,rep=rep,method=method,total_s=elapsed)
 write.csv(do.call(rbind,rows),paste0(base,'-Rtimes.csv'),row.names=FALSE)
 cat('DONE',case,seed,rep,method,elapsed,'\n');flush.console()
}
writeLines(capture.output(sessionInfo()),paste0(base,'-Rsession.txt'))
