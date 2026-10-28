suppressPackageStartupMessages(library(rrpack))
out <- commandArgs(TRUE)[1]
writeLines(as.character(packageVersion("rrpack")), file.path(out,"rrpack-version.txt"))
for (case in 1:3) {
 X <- as.matrix(read.csv(file.path(out,paste0("X",case,".csv")),header=FALSE))
 Y <- as.matrix(read.csv(file.path(out,paste0("Y",case,".csv")),header=FALSE))
 edf <- rrr(Y,X,maxrank=3,df.type="exact",modstr=list(lambda=1:3))
 write.table(edf$df.exact,file.path(out,paste0("df-",case,".csv")),sep=",",row.names=FALSE,col.names=FALSE)
 for (r in 1:3) {
  s <- rrr(Y,X,maxrank=r,df.type="naive",modstr=list(lambda=r))
  write.table(coef(s),file.path(out,paste0("rrr-",case,"-",r,".csv")),sep=",",row.names=FALSE,col.names=FALSE)
 }
 if (case==3) next
 ini <- rrpack:::rrr.fit(Y,X,nrank=2,coefSVD=TRUE)
 A <- ini$coefSVD$u %*% diag(ini$coefSVD$d)
 V <- ini$coefSVD$v
 for (lam in c(0.01,0.05,0.2)) for (adaptive in c(FALSE,TRUE)) {
  w <- if (adaptive) pmax(sqrt(rowSums(ini$coef^2)),sqrt(.Machine$double.eps))^(-2) else rep(1,ncol(X))
  s <- srrr(Y,X,nrank=2,A0=A,V0=V,modstr=list(lamA=nrow(X)*lam,nlam=1,WA=w),control=list(epsilon=1e-10,maxit=1000,inner.eps=1e-11,inner.maxit=10000))
  tag <- paste0("srrr-",case,"-",lam,"-",tolower(adaptive))
  write.table(coef(s),file.path(out,paste0(tag,".csv")),sep=",",row.names=FALSE,col.names=FALSE)
 }
 # rrpack 0.1-14 rssvd.eea maps BIC to column 2 and BICP to column 1,
 # while ic.path stores BIC in column 1. Request its BICP label, and verify
 # the actual BIC selection below rather than reproducing the label bug.
 s <- rssvd(Y,X,nrank=2,ic.type="BICP",control=list(nlambda=10,epsilon=1e-10,innerEpsilon=1e-11,maxit=1000,innerMaxit=10000,minLambda=1e-5))
 expected <- matrix(0,ncol(X),ncol(Y))
 for (k in 1:2) {
  j <- which.min(s$ic.path[,k,1])
  expected <- expected + s$U.path[,j,k] %o% s$V.path[,j,k] * s$D.path[j,k]
 }
 stopifnot(max(abs(coef(s)-expected)) < 1e-8)
 write.table(coef(s),file.path(out,paste0("rssvd-",case,".csv")),sep=",",row.names=FALSE,col.names=FALSE)
}
