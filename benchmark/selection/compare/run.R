args <- commandArgs(TRUE); mode <- args[1]; rep <- as.integer(args[2])
root <- Sys.getenv('RUN'); out <- file.path(root,'results')
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/Rlib',.libPaths()))
library(SLOPE);library(knockoff);library(stabs);library(glmnet)
set.seed(8500+rep)
readcase <- function(con) {
  h <- readBin(con,'numeric',6,size=8,endian='little')
  if(!length(h)) return(NULL)
  n<-h[1];p<-h[2];k<-h[3];sc<-h[4];seed<-h[5];B<-h[6]
  X<-matrix(readBin(con,'numeric',n*p,size=8,endian='little'),n,p)
  y<-readBin(con,'numeric',n,size=8,endian='little')
  Z<-matrix(readBin(con,'numeric',n*p,size=8,endian='little'),n,p)
  rows<-matrix(readBin(con,'integer',n*B,size=4,endian='little'),n%/%2,2*B)
  list(X=X,y=y,Z=Z,rows=rows,n=n,p=p,k=k,sc=sc,seed=seed,B=B)
}
sigma <- function(p,rho) outer(seq_len(p),seq_len(p),function(i,j) rho^abs(i-j))
bh <- function(p,n) qnorm(1-0.1*seq_len(p)/(2*p))/sqrt(n)
slopefit <- function(X,y,lambda,factors,solver='auto') {
  fit<-SLOPE::SLOPE(X,y,lambda=lambda,alpha=factors,gamma=1,intercept=FALSE,
    center='none',scale='none',tol=if(solver=='fista') 1e-14 else 1e-8,max_passes=1000000,threads=1,
    tol_dev_ratio=1,tol_dev_change=0,max_variables=ncol(X)+1,solver=solver)
  b<-as.matrix(coef(fit,intercept=FALSE))
  stopifnot(ncol(b)==length(factors))
  b
}
lassofit <- function(X,y,lambda) {
  fit<-glmnet::glmnet(X,y,lambda=lambda,intercept=FALSE,standardize=FALSE,
    thresh=1e-14,maxit=1000000)
  stopifnot(fit$jerr==0)
  as.numeric(coef(fit))[-1]
}
wdifference <- function(b) {p<-length(b)%/%2;abs(b[seq_len(p)])-abs(b[p+seq_len(p)])}
selw <- function(W) which(W>=knockoff::knockoff.threshold(W,fdr=0.1,offset=1))
fixedfit <- function(x,y,q,lambda,...) {
  b<-abs(lassofit(x,y,lambda));s<-rep(FALSE,ncol(x))
  s[head(order(b,decreasing=TRUE),min(q,sum(b>0)))]<-TRUE
  list(selected=s)
}
cpss <- function(c,cap,lambda=0.06,path=FALSE) {
  folds<-matrix(0,c$n,c$B)
  for(j in seq_len(c$B)) folds[c$rows[,2*j-1],j]<-1
  if(path) {
    stabs::stabsel(c$X,c$y,fitfun=stabs::glmnet.lasso,cutoff=0.8,q=cap,folds=folds,B=c$B,
      assumption='none',sampling.type='SS',papply=lapply,verbose=FALSE,
      args.fitfun=list(intercept=FALSE,standardize=FALSE,thresh=1e-14,maxit=1000000))
  } else {
    stabs::stabsel(c$X,c$y,fitfun=fixedfit,cutoff=0.8,q=cap,folds=folds,B=c$B,
      assumption='none',sampling.type='SS',papply=lapply,verbose=FALSE,
      args.fitfun=list(lambda=lambda))
  }
}
write_mat <- function(x,name) write.table(as.matrix(x),file.path(out,name),sep=',',row.names=FALSE,col.names=FALSE,quote=FALSE)
record <- function(io,c,method,s,lambda=NaN,cap=0,bound=NaN) {
  V<-sum(s>c$k);R<-length(s);power<-if(c$k>0) (R-V)/c$k else NaN
  cat(paste(c(c$sc,c$seed,c$n,c$p,c$k,method,lambda,cap,bound,R,V,V/max(1,R),power,paste(s,collapse=';')),collapse=','),'\n',file=io,sep='')
  flush(io)
}
timed <- function(io,case,engine,f) {
  warm<-system.time(f())[['elapsed']];gc()
  ncalls<-max(1,min(20,ceiling(0.02/max(warm,1e-6))))
  for(inner in 1:3) {
    t<-system.time(for(j in seq_len(ncalls)) v<-f())[['elapsed']]/ncalls
    cat(paste(c(rep,case,engine,inner,t,NA),collapse=','),'\n',file=io,sep='');flush(io)
  }
  v
}
if(mode=='benchmark') {
  io<-file(file.path(out,paste0('timing-R-',rep,'.csv')),'w')
  cat('rep,case,engine,inner,seconds,bytes\n',file=io)
  bi<-file(file.path(root,'inputs','bench.bin'),'rb')
  repeat {
    c<-readcase(bi);if(is.null(c)) break
    X<-if(c$sc==4) Matrix::Matrix(c$X,sparse=TRUE) else c$X
    factors<-scan(file.path(out,paste0('factors-',c$sc,'.csv')),quiet=TRUE)
    solvers<-strsplit(Sys.getenv('R_SOLVERS','auto'),',',fixed=TRUE)[[1]]
    if(rep%%2==0) solvers<-rev(solvers)
    for(solver in solvers) {
      b<-timed(io,paste0('slope-',c$sc),paste0('R-SLOPE-',solver),function() slopefit(X,c$y,bh(c$p,c$n),factors,solver))
      write_mat(b,paste0('coef-',rep,'-slope-',c$sc,'-R-',solver,'.csv'))
    }
    if(c$sc %in% c(1,2)) {
      S<-sigma(c$p,0.3)
      timed(io,paste0('gaussian-',c$sc),'R-knockoff',function() knockoff::create.gaussian(c$X,rep(0,c$p),S,method='equi'))
      if(c$sc==1) {
        F<-sweep(c$X,2,sqrt(colSums(c$X^2)),'/')
        fixed<-timed(io,'fixed','R-knockoff',function() knockoff::create.fixed(F,method='equi'))
        G<-crossprod(F);sfix<-knockoff::create.solve_equi(G)
        write.csv(data.frame(rep=rep,engine='R-knockoff',max_gram_error=max(abs(crossprod(fixed$Xk)-G)),max_cross_gram_error=max(abs(crossprod(F,fixed$Xk)-(G-diag(sfix))))),
          file.path(out,paste0('fixed-validity-R-',rep,'.csv')),row.names=FALSE)
        ki<-file(file.path(out,'bench-knockoffs.bin'),'rb')
        K<-matrix(readBin(ki,'numeric',c$n*c$p,size=8,endian='little'),c$n,c$p);close(ki)
        A<-cbind(c$X,K)
        b<-timed(io,'joint-lasso','R-glmnet',function() lassofit(A,c$y,0.04))
        write_mat(b,paste0('coef-',rep,'-lasso-R-glmnet.csv'))
        freq<-timed(io,'cpss','R-stabs',function() cpss(c,10,lambda=0.08)$max)
        write_mat(freq,paste0('freq-',rep,'-R-stabs.csv'))
      }
    }
  }
  close(bi);close(io)
  set.seed(8400+rep);p<-30;n<-20000;S<-sigma(p,0.8);s<-knockoff::create.solve_equi(S)
  C<-2*diag(s)-diag(s)%*%solve(S,diag(s))
  Z<-matrix(rnorm(n*p),n,p);set.seed(8400+rep)
  K<-knockoff::create.gaussian(matrix(0,n,p),rep(0,p),S,method='equi')
  root<-qr.solve(Z,K)
  write.csv(data.frame(rep=rep,engine='R-knockoff',relative_covariance_error=norm(cov(K)-C,'F')/norm(C,'F'),implied_covariance_error=norm(crossprod(root)-C,'F')/norm(C,'F')),
    file.path(out,paste0('validity-R-',rep,'.csv')),row.names=FALSE)
  writeLines(capture.output(sessionInfo()),file.path(out,paste0('R-session-',rep,'.txt')))
} else if(mode=='mc') {
  io<-file(file.path(out,paste0('mc-R-',rep,'.csv')),'w')
  cat('scenario,seed,n,p,k,method,lambda,cap,bound,selected,false,fdp,power,selected_ids\n',file=io)
  bi<-file(file.path(root,'inputs',paste0('mc-',rep,'.bin')),'rb')
  ki<-file(file.path(root,'inputs',paste0('shared-',rep,'.bin')),'rb')
  repeat {
    c<-readcase(bi);if(is.null(c)) break
    b<-slopefit(c$X,c$y,bh(c$p,c$n),1)
    record(io,c,'slope-R',which(abs(b)>1e-6))
    for(method in c('equi','mvr','maxent')) {
      K<-matrix(readBin(ki,'numeric',c$n*c$p,size=8,endian='little'),c$n,c$p)
      W<-wdifference(lassofit(cbind(c$X,K),c$y,0.04))
      record(io,c,paste0('mx-',method,'-R'),selw(W),lambda=0.04)
    }
    if(c$n>=2*c$p) {
      K<-matrix(readBin(ki,'numeric',c$n*c$p,size=8,endian='little'),c$n,c$p)
      record(io,c,'fx-R',selw(wdifference(lassofit(cbind(c$X,K),c$y,0.04))),lambda=0.04)
    }
    rho<-if(c$sc==1) 0 else if(c$sc==3) 0.8 else 0.4
    set.seed(c$seed+19)
    K<-knockoff::create.gaussian(c$X,rep(0,c$p),sigma(c$p,rho),method='equi')
    record(io,c,'mx-native-equi-R',selw(wdifference(lassofit(cbind(c$X,K),c$y,0.04))),lambda=0.04)
    for(cap in c(10,20)) {
      st<-cpss(c,cap)
      record(io,c,paste0('cpss-cap',cap,'-R'),which(st$max>=0.8),lambda=0.06,cap=cap,bound=st$PFER)
    }
    st<-cpss(c,10,path=TRUE)
    record(io,c,'cpss-path10-R',which(st$max>=0.8),cap=10,bound=st$PFER)
  }
  close(bi);close(ki);close(io)
} else stop('unknown mode')
cat('DONE',mode,rep,'\n')
