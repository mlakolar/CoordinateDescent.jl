args<-commandArgs(TRUE);case<-as.integer(args[1]);rep<-as.integer(args[2]);root<-Sys.getenv('HDM_KNOCKOFF_RUNTIME')
inputdir<-file.path(root,paste0('candidate-case-',case,'-rep-',rep));out<-file.path(root,paste0('R-case-',case,'-rep-',rep))
stopifnot(!dir.exists(out));dir.create(out)
library(knockoff);stopifnot(as.character(packageVersion('knockoff'))=='0.3.6')
readi64<-function(con,n){bytes<-readBin(con,'raw',8*n);stopifnot(length(bytes)==8*n);vapply(seq_len(n),function(i)sum(as.integer(bytes[(8*i-7):(8*i)])*256^(0:7)),numeric(1))}
io<-file(file.path(out,'timings.csv'),'w');cat('case,rep,input,seed,payload,round,seconds\n',file=io)
checks<-file(file.path(out,'checks.csv'),'w');cat('input,round,valid,gram_error,cross_error\n',file=checks)
for(input in seq_len(3)){
 src<-file(file.path(inputdir,paste0('input-',input,'.bin')),'rb');h<-readi64(src,3);n<-h[1];p<-h[2];seed<-h[3]
 X<-matrix(readBin(src,'numeric',n*p,size=8,endian='little'),n,p);y<-readBin(src,'numeric',n,size=8,endian='little')
 Z<-matrix(readBin(src,'numeric',n*p,size=8,endian='little'),n,p);S<-matrix(readBin(src,'numeric',p*p,size=8,endian='little'),p,p);close(src)
 if(case==5){
  src<-file(file.path(inputdir,paste0('unitnorm-',input,'.bin')),'rb');F<-matrix(readBin(src,'numeric',n*p,size=8,endian='little'),n,p);close(src)
  call<-function()knockoff::create.fixed(F,method='equi');payload<-'construct-unitnorm'
 }else{
  # Includes RNG setup and draw, as does the paired native construct-rng call.
  call<-function(){set.seed(as.integer(seed%%2000000000));knockoff::create.gaussian(X,rep(0,p),S,method='equi')};payload<-'construct-rng'
 }
 if(case<5&&input==1){
  diagnostic_seed<-as.integer(seed%%2000000000)
  set.seed(diagnostic_seed);Zr<-matrix(rnorm(p*p),p,p)
  set.seed(diagnostic_seed);K0<-knockoff::create.gaussian(matrix(0,p,p),rep(0,p),S,method='equi')
  set.seed(diagnostic_seed);K1<-knockoff::create.gaussian(diag(p),rep(0,p),S,method='equi')
  D<-S%*%(diag(p)-(K1-K0));d<-diag(D);C<-2*diag(d)-diag(d)%*%solve(S,diag(d));factor<-solve(Zr,K0)
  diagonal_error<-norm(D-diag(d),'F')/max(1,norm(D,'F'))
  root_error<-norm(crossprod(factor)-C,'F')/max(1,norm(C,'F'))
  mineig<-min(eigen(2*S-diag(d),symmetric=TRUE,only.values=TRUE)$values)
  write.csv(data.frame(root_error=root_error,diagonal_error=diagonal_error,joint_min=mineig),file.path(out,'root-identity.csv'),row.names=FALSE)
  stopifnot(root_error<1e-8,diagonal_error<1e-8,mineig> -1e-8,min(d)> -1e-12)
 }
 warm<-call();batch<-1
 # Use a short matched-clock batch when R's elapsed clock cannot resolve one call.
 elapsed<-system.time(call())[['elapsed']];batch<-max(1,min(20,ceiling(.02/max(elapsed,1e-6))))
 for(round in seq_len(3)){
  gc();start<-proc.time()[['elapsed']]
  for(i in seq_len(batch))value<-call()
  seconds<-(proc.time()[['elapsed']]-start)/batch
  cat(paste(c(case,rep,input,seed,payload,round,seconds),collapse=','),'\n',file=io,sep='');flush(io)
  if(case==5){
   K<-value$Xk;G<-crossprod(F);d<-knockoff::create.solve_equi(G)
   gram<-max(abs(crossprod(K)-G));cross<-max(abs(crossprod(F,K)-(G-diag(d))));valid<-gram<1e-8&&cross<1e-8
  }else{gram<-NA;cross<-NA;valid<-all(is.finite(value))}
  cat(paste(c(input,round,valid,gram,cross),collapse=','),'\n',file=checks,sep='');flush(checks);stopifnot(valid)

 }
 if(case==5){
  K<-warm$Xk;G<-crossprod(F);d<-knockoff::create.solve_equi(G)
  gram<-max(abs(crossprod(K)-G));cross<-max(abs(crossprod(F,K)-(G-diag(d))))
  valid<-gram<1e-8&&cross<1e-8
 }else{gram<-NA;cross<-NA;valid<-all(is.finite(warm))}
 cat(paste(c(input,0,valid,gram,cross),collapse=','),'\n',file=checks,sep='');flush(checks);stopifnot(valid)
}
close(io);close(checks);writeLines(capture.output(sessionInfo()),file.path(out,'session.txt'))
cat('R_RUNTIME_DONE',case,rep,'\n')
