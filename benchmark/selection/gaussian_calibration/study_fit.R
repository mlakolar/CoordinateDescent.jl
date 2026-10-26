args<-commandArgs(TRUE);sc<-as.integer(args[1]);shard<-as.integer(args[2])
root<-Sys.getenv('HDM_STUDY');input<-file.path(root,'inputs',paste0('sc',sc,'-shard',shard,'.bin'))
out<-file.path(root,'fits',paste0('R-sc',sc,'-shard',shard));dir.create(dirname(out),showWarnings=FALSE,recursive=TRUE)
stopifnot(!file.exists(paste0(out,'.done')))
library(SLOPE)
stopifnot(as.character(packageVersion('SLOPE'))=='2.1.1')
src<-file(input,'rb');dst<-file(paste0(out,'.bin'),'wb');status<-file(paste0(out,'.csv'),'w')
cat('seed,method,seconds,reported_converged,iterations,residual,error\n',file=status)
# Float64 headers are used in fit records. Inputs use little-endian Int64,
# read with numeric size8 then reinterpret via a small byte decoder.
readi64<-function(con,n) {
  bytes<-readBin(con,'raw',8*n)
  if(!length(bytes)) return(numeric())
  stopifnot(length(bytes)==8*n)
  vapply(seq_len(n),function(i) sum(as.integer(bytes[(8*i-7):(8*i)])*256^(0:7)),numeric(1))
}
writei64<-function(x,con) for(a in x) writeBin(as.raw(floor(a/256^(0:7))%%256),con)
repeat {
  h<-readi64(src,8);if(!length(h)) break
  n<-h[1];p<-h[2];seed<-h[5];intercept<-as.logical(h[6]);sigma<-readBin(src,'numeric',1,size=8,endian='little')
  X<-matrix(readBin(src,'numeric',n*p,size=8,endian='little'),n,p)
  y<-readBin(src,'numeric',n,size=8,endian='little');truth<-readBin(src,'numeric',p,size=8,endian='little')
  W<-matrix(readBin(src,'numeric',p*3,size=8,endian='little'),p,3)
  native<-sigma/sqrt(n)*SLOPE::regularizationWeights(p,type='gaussian',q=.1,n=n-as.integer(intercept))
  W<-cbind(W,native);B<-matrix(NaN,p+1,4)
  xb<-if(intercept) colMeans(X) else rep(0,p);yb<-if(intercept) mean(y) else 0
  Z<-sweep(X,2,xb);v<-y-yb
  for(j in seq_len(4)) {
    method<-c('bh','paper','julia','R-native')[j];err<-'';ok<-FALSE;start<-proc.time()[['elapsed']]
    tryCatch({
      z<-SLOPE::SLOPE(Z,v,lambda=W[,j],alpha=1,gamma=1,intercept=FALSE,center='none',scale='none',
        tol=1e-8,max_passes=1000000,threads=1,tol_dev_ratio=1,tol_dev_change=0,max_variables=p+1,solver='auto')
      b<-as.matrix(coef(z,intercept=FALSE));stopifnot(ncol(b)==1)
      B[,j]<-c(b,yb-sum(xb*b));ok<-TRUE
    },error=function(e) err<<-gsub('[,\n]',';',conditionMessage(e)))
    cat(paste(c(seed,method,proc.time()[['elapsed']]-start,ok,-1,NaN,err),collapse=','),'\n',file=status,sep='');flush(status)
  }
  writei64(c(seed,p,4),dst);writeBin(as.double(W),dst,size=8,endian='little');writeBin(as.double(B),dst,size=8,endian='little');flush(dst)
}
close(src);close(dst);close(status)
writeLines(c(capture.output(sessionInfo()),paste('input',input),paste('OMP_NUM_THREADS',Sys.getenv('OMP_NUM_THREADS'))),paste0(out,'.session.txt'))
writeLines('complete',paste0(out,'.done'))
