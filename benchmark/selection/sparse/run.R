args<-commandArgs(TRUE);rep<-as.integer(args[1]);mode<-args[2]
root<-Sys.getenv('RUN');out<-file.path(root,Sys.getenv('RESULTS_DIR','results'))
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib',.libPaths()))
library(SLOPE);library(Matrix)
readcase<-function(id) {
 con<-file(file.path(root,'inputs',paste0('case-',id,'.bin')),'rb');on.exit(close(con))
 # All counts in this bounded campaign fit exactly in 32 bits. Decode the
 # little-endian Int64 headers/indices by selecting their low words.
 ints<-function(n) readBin(con,'integer',2*n,size=4,endian='little')[seq(1,2*n,by=2)]
 h<-ints(7);n<-h[1];p<-h[2];nz<-h[4];L<-h[5]
 if(h[3]==1) {
   ptr<-ints(p+1)-1L;rows<-ints(nz)-1L;val<-readBin(con,'numeric',nz,size=8,endian='little')
   X<-new('dgCMatrix',Dim=as.integer(c(n,p)),p=ptr,i=rows,x=val)
 } else X<-matrix(readBin(con,'numeric',n*p,size=8,endian='little'),n,p)
 list(X=X,y=readBin(con,'numeric',n,size=8,endian='little'),lambda=readBin(con,'numeric',p,size=8,endian='little'),
      factors=readBin(con,'numeric',L,size=8,endian='little'),intercept=h[6]==1,std=h[7]==1)
}
fit<-function(c) {
 f<-SLOPE::SLOPE(c$X,c$y,lambda=c$lambda,alpha=c$factors,gamma=1,
   intercept=c$intercept,center=if(c$intercept) 'mean' else 'none',scale=if(c$std) 'sd' else 'none',
   tol=1e-9,max_passes=1000000,threads=1,tol_dev_ratio=1,tol_dev_change=0,
   max_variables=ncol(c$X)+1,solver='auto')
 b<-as.matrix(coef(f,intercept=FALSE));stopifnot(ncol(b)==length(c$factors))
 b0<-if(c$intercept) as.matrix(coef(f,intercept=TRUE))[1,] else rep(0,ncol(b))
 rbind(b0,b)
}
ids<-if(mode=='memory') as.integer(args[3]) else as.integer(strsplit(Sys.getenv('CASES','1,2,3,4,5,6,7,8,9,10,11'),',',fixed=TRUE)[[1]])
dir.create(out,recursive=TRUE,showWarnings=FALSE)
fname<-paste0(mode,'-R-',rep,if(mode=='memory') paste0('-',ids[1]) else '', '.csv')
io<-file(file.path(out,fname),'w');cat('rep,case,engine,inner,seconds,bytes,relative_residual,design_bytes,workspace_bytes\n',file=io)
for(id in ids) {
 c<-readcase(id);fit(c);warm<-system.time(fit(c))[['elapsed']];gc()
 ncalls<-if(mode=='memory') 1 else max(1,min(20,ceiling(0.1/max(warm,1e-6))))
 for(inner in seq_len(if(mode=='memory') 1 else 3)) {
   t<-system.time(for(j in seq_len(ncalls)) b<-fit(c))[['elapsed']]/ncalls
   cat(paste(c(rep,id,'R',inner,t,NA,NA,NA,NA),collapse=','),'\n',file=io,sep='');flush(io)
 }
 write.table(b,file.path(out,paste0('coef-',rep,'-',id,'-R.csv')),sep=',',row.names=FALSE,col.names=FALSE,quote=FALSE)
 cat('DONE R',mode,'rep=',rep,'case=',id,'\n');flush.console();gc()
}
close(io)
writeLines(capture.output(sessionInfo()),file.path(out,paste0('R-session-',rep,'.txt')))
