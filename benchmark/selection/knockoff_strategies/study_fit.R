args<-commandArgs(TRUE);sc<-as.integer(args[1]);shard<-as.integer(args[2])
root<-Sys.getenv('HDM_KNOCKOFF_STUDY');out<-file.path(root,paste0('scenario-',sc,'-shard-',shard))
library(glmnet)
stopifnot(as.character(packageVersion('glmnet'))=='5.1')
stopifnot(!file.exists(file.path(out,'R-fits.bin')))
readi64<-function(con,n) {
 bytes<-readBin(con,'raw',8*n);if(!length(bytes))return(numeric())
 stopifnot(length(bytes)==8*n)
 vapply(seq_len(n),function(i)sum(as.integer(bytes[(8*i-7):(8*i)])*256^(0:7)),numeric(1))
}
writei64<-function(x,con)for(a in x)writeBin(as.raw(floor(a/256^(0:7))%%256),con)
src<-file(file.path(out,'R-inputs.bin'),'rb');dst<-file(file.path(out,'R-fits.bin'),'wb');status<-file(file.path(out,'R-status.csv'),'w')
cat('seed,strategy,success,jerr,error\n',file=status)
repeat {
 h<-readi64(src,8);if(!length(h))break
 n<-h[1];p<-h[2];seed<-h[7];j<-h[8]
 A<-matrix(readBin(src,'numeric',n*2*p,size=8,endian='little'),n,2*p)
 y<-readBin(src,'numeric',n,size=8,endian='little');b<-rep(NaN,2*p);ok<-FALSE;err<-'';jerr<-NA
 tryCatch({
  fit<-glmnet::glmnet(A,y,lambda=.04,intercept=FALSE,standardize=FALSE,thresh=1e-14,maxit=1000000)
  jerr<-fit$jerr;b<-as.numeric(coef(fit))[-1];ok<-jerr==0
 },error=function(e)err<<-gsub('[,\n]',';',conditionMessage(e)))
 writei64(c(seed,p,j),dst);writeBin(as.double(b),dst,size=8,endian='little');flush(dst)
 cat(paste(c(seed,c('equicorrelated','mvr','maxent')[j],ok,jerr,err),collapse=','),'\n',file=status,sep='');flush(status)
}
close(src);close(dst);close(status)
writeLines(c(capture.output(sessionInfo()),paste('OMP_NUM_THREADS',Sys.getenv('OMP_NUM_THREADS'))),file.path(out,'R-session.txt'))
cat('R_COMPARATOR_DONE',sc,shard,'\n')
