suppressPackageStartupMessages(library(rrpack))
replication <- as.integer(commandArgs(TRUE)[1])
input <- paste0("inputs/medium-",replication)
out <- paste0("results/adaptive/medium-",replication,"/r")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
readmat <- function(name,nr,nc) {
 con <- file(file.path(input,paste0(name,".bin")),"rb")
 on.exit(close(con));matrix(readBin(con,"double",nr*nc,size=8,endian="little"),nr,nc)
}
n <-300;p<-50;q<-100;r<-3
X<-readmat("X",n,p);Y<-readmat("Y",n,q)
X<-sweep(X,2,colMeans(X));Y<-sweep(Y,2,colMeans(Y))
A0<-readmat("A0",p,r);V0<-readmat("V0",q,r)
w<-as.vector(readmat("adaptive_weights",p,1))
lambda_max<-scan(file.path(input,"adaptive_max.txt"),quiet=TRUE)
normf<-function(x)sqrt(sum(x*x))
fit<-function(frac)rrpack::srrr(Y,X,nrank=r,A0=A0,V0=V0,
 modstr=list(lamA=n*frac*lambda_max,nlam=1,WA=w),
 control=list(epsilon=1e-8,maxit=1000,inner.eps=1e-9,inner.maxit=10000))
fit<-compiler::cmpfun(fit)
invisible(fit(.1))
file<-file.path(out,"adaptive.csv")
writeLines("phase,language,case,rep,family,solver,mode,frac,round,seconds,iterations,objective,gap,residual,status,native_converged,rss_peak_kb,rss_baseline_kb,detail",file)
for(rd in 1:3)for(frac in c(.5,.1,.02)) {
 t<-Sys.time();z<-fit(frac);secs<-as.numeric(difftime(Sys.time(),t,units="secs"))
 A<-matrix(z$A.path[1,,],p,r);V<-matrix(z$V.path[1,,],q,r);B<-coef(z)
 E<-X%*%A%*%t(V)-Y;G<-crossprod(X,E)%*%V/n;an<-sqrt(rowSums(A*A));lambda<-frac*lambda_max
 errs<-pmax(0,sqrt(rowSums(G*G))-lambda*w);nz<-an>1e-12
 errs[nz]<-sqrt(rowSums((G[nz,,drop=FALSE]+A[nz,,drop=FALSE]*(lambda*w[nz]/an[nz]))^2))
 GV<-crossprod(E,X)%*%A/n;tangent<-GV-V%*%((crossprod(V,GV)+crossprod(GV,V))/2)
 residual<-max(max(errs)/max(1,normf(crossprod(X,Y)%*%V/n)),
              normf(tangent)/max(1,normf(crossprod(Y,X)%*%A/n)))
 obj<-sum(E*E)/(2*n)+lambda*sum(w*an)
 status<-if(residual<=1e-6)"passed" else "accuracy_failed"
 write(paste(c("adaptive","r","medium",replication,"sparse_adaptive","r_srrr_adaptive","fit",
  frac,rd,secs,-1,obj,NaN,residual,status,NA,NA,NA,""),collapse=","),file=file,append=TRUE)
 if(rd==1)for(tag in c("bin","A","V")) {
  con<-file(file.path(out,paste0("r_srrr_adaptive-",frac,".",tag)),"wb")
  writeBin(as.double(if(tag=="bin")B else if(tag=="A")A else V),con,size=8,endian="little");close(con)
 }
}
Rprof(file.path(out,"profile-r_srrr_adaptive.Rprof"),interval=.001)
t<-proc.time()[3];repeat{z<-fit(.1);if(proc.time()[3]-t>.25)break}
Rprof(NULL)
capture.output(summaryRprof(file.path(out,"profile-r_srrr_adaptive.Rprof")),
 file=file.path(out,"profile-r_srrr_adaptive.txt"))
