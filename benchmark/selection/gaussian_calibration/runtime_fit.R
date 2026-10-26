args<-commandArgs(TRUE);sc<-as.integer(args[1]);rep<-as.integer(args[2])
root<-Sys.getenv('HDM_STUDY');input<-file.path(root,'inputs',paste0('sc',sc,'-shard',rep,'.bin'))
out<-file.path(root,'fits',paste0('R-sc',sc,'-shard',rep));dir.create(paste0(out,'-profiles'),showWarnings=FALSE,recursive=TRUE)
stopifnot(!file.exists(paste0(out,'.done')))
library(SLOPE);stopifnot(as.character(packageVersion('SLOPE'))=='2.1.1')
readi64<-function(con,n) {
  b<-readBin(con,'raw',8*n);if(!length(b)) return(numeric());stopifnot(length(b)==8*n)
  vapply(seq_len(n),function(i) sum(as.integer(b[(8*i-7):(8*i)])*256^(0:7)),numeric(1))
}
writei64<-function(x,con) for(a in x) writeBin(as.raw(floor(a/256^(0:7))%%256),con)
src<-file(input,'rb');dst<-file(paste0(out,'.bin'),'wb');records<-file(paste0(out,'.records.csv'),'w');timings<-file(paste0(out,'.timings.csv'),'w')
cat('seed,method,payload,setting,round,ncolumns,reported_converged\n',file=records)
cat('seed,method,payload,setting,round,seconds,bytes,gctime\n',file=timings)
repeat {
  h<-readi64(src,8);if(!length(h)) break
  n<-h[1];p<-h[2];seed<-h[5];dataset<-h[7];sigma<-readBin(src,'numeric',1,size=8,endian='little')
  X<-matrix(readBin(src,'numeric',n*p,size=8,endian='little'),n,p);y<-readBin(src,'numeric',n,size=8,endian='little')
  truth<-readBin(src,'numeric',p,size=8,endian='little');W<-matrix(readBin(src,'numeric',p*3,size=8,endian='little'),p,3)
  if(sc==4) X<-Matrix::Matrix(X,sparse=TRUE)
  methods<-((seq_len(3)+rep+dataset-1)%%3)+1
  for(j in methods) for(payload in c('fit','path')) {
    w<-W[,j];amax<-max(cumsum(sort(abs(as.numeric(crossprod(X,y))/n),decreasing=TRUE))/cumsum(w))
    factors<-if(payload=='fit') 1 else exp(seq(log(amax),log(.2*amax),length.out=20))
    callfit<-function() SLOPE::SLOPE(X,y,lambda=w,alpha=factors,gamma=1,intercept=FALSE,center='none',scale='none',
      tol=1e-10,max_passes=1000000,threads=1,tol_dev_ratio=1,tol_dev_change=0,max_variables=p+1,solver='auto')
    warm<-callfit()
    for(round in seq_len(3)) {
      gc();t0<-as.numeric(Sys.time());z<-callfit();elapsed<-as.numeric(Sys.time())-t0
      B<-as.matrix(coef(z,intercept=FALSE));stopifnot(ncol(B)==length(factors))
      B<-rbind(B,rep(0,ncol(B)));actual_weights<-w%o%factors
      writei64(c(seed,p,ncol(B)),dst);writeBin(as.double(actual_weights),dst,size=8,endian='little');writeBin(as.double(B),dst,size=8,endian='little');flush(dst)
      cat(paste(c(seed,c('bh','paper','julia')[j],payload,'public',round,ncol(B),'unavailable'),collapse=','),'\n',file=records,sep='');flush(records)
      cat(paste(c(seed,c('bh','paper','julia')[j],payload,'public',round,elapsed,NA,NA),collapse=','),'\n',file=timings,sep='');flush(timings)
    }
    prefix<-file.path(paste0(out,'-profiles'),paste(dataset,c('bh','paper','julia')[j],payload,'public',sep='-'))
    Rprof(paste0(prefix,'.cpu.txt'),interval=.001)
    for(i in seq_len(10)) invisible(callfit())
    Rprof(NULL)
    Rprofmem(paste0(prefix,'.alloc.txt'),threshold=16384);invisible(callfit());Rprofmem(NULL)
  }
}
close(src);close(dst);close(records);close(timings)
writeLines(capture.output(sessionInfo()),paste0(out,'.session.txt'));writeLines('complete',paste0(out,'.done'))
