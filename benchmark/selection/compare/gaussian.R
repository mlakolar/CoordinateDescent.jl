# Load shared readers and fit wrappers; the source's mc section only defines
# functions when mode is set to helpers, without running the original campaign.
args<-commandArgs(TRUE);rep<-as.integer(args[2])
lines<-readLines(file.path(Sys.getenv('RUN'),'source','benchmark','selection','compare','run.R'))
start<-grep("^if\\(mode=='benchmark'\\)",lines)[1]
eval(parse(text=lines[1:(start-1)]))
set.seed(8600+rep)
io<-file(file.path(out,paste0('gaussian-mc-R-',rep,'.csv')),'w')
cat('scenario,seed,n,p,k,method,lambda,cap,bound,selected,false,fdp,power,selected_ids\n',file=io)
di<-file(file.path(out,paste0('gaussian-weight-difference-',rep,'.csv')),'w')
cat('scenario,seed,max_weight_difference\n',file=di)
bi<-file(file.path(root,'inputs',paste0('mc-',rep,'.bin')),'rb')
wi<-file(file.path(root,'inputs',paste0('gaussian-weights-',rep,'.bin')),'rb')
repeat {
  c<-readcase(bi);if(is.null(c)) break
  lambda<-readBin(wi,'numeric',c$p,size=8,endian='little')
  b<-slopefit(c$X,c$y,lambda,1)
  record(io,c,'gaussian-R-common',which(abs(b)>1e-6))
  fit<-SLOPE::SLOPE(c$X,c$y,lambda='gaussian',alpha=1/sqrt(c$n),gamma=1,intercept=FALSE,
    center='none',scale='none',tol=1e-8,max_passes=100000,threads=1,
    tol_dev_ratio=1,tol_dev_change=0,max_variables=c$p+1,solver='auto')
  b<-as.matrix(coef(fit,intercept=FALSE))
  stopifnot(ncol(b)==1)
  record(io,c,'gaussian-R-native',which(abs(b)>1e-6))
  cat(paste(c(c$sc,c$seed,max(abs(as.numeric(fit$lambda)*fit$alpha[1]-lambda))),collapse=','),'\n',file=di,sep='')
}
close(bi);close(wi);close(io);close(di)
cat('GAUSSIAN_DONE R',rep,'\n')
