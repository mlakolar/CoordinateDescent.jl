# Generate identical inputs for all solvers, inside a Discovery allocation.
args <- commandArgs(TRUE); out <- args[1]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
put <- function(a, f) write.table(a, f, sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)
cases <- list()
add <- function(id, family, n, p, order, grid, path, lo, hi, genlasso) {
  cases[[length(cases)+1]] <<- data.frame(id, family, n, p, order, grid, path, lo, hi, genlasso)
}
for (n in c(1000,10000,100000)) for (k in 0:2) for (lam in c(.1,1,10))
  add(sprintf("signal-n%d-k%d-l%g",n,k,lam),"signal",n,n,k,"unit",0,lam,lam,n==1000)
for (k in 0:2) for (lam in c(.1,1,10))
  add(sprintf("irregular-n10000-k%d-l%g",k,lam),"signal",10000,10000,k,"irregular",0,lam,lam,FALSE)
for (n in c(1000,10000)) for (k in 0:2)
  add(sprintf("signal-path-n%d-k%d",n,k),"signal",n,n,k,"unit",50,.1,10,n==1000)
for (p in c(40,100)) for (f in c("fused","graph","trend")) {
  n <- if(p==40) 200 else 1000
  for (lam in c(.01,.1)) add(sprintf("%s-p%d-l%g",f,p,lam),f,n,p,1,"unit",0,n*lam,n*lam,TRUE)
  add(sprintf("%s-path-p%d",f,p),f,n,p,1,"unit",50,n*.003,n*.1,TRUE)
}
cases <- do.call(rbind,cases)
cases <- cases[(cases$family=="signal" & cases$order==2 & cases$n<=10000) |
  (cases$p==100 & cases$family=="fused" & (cases$path>0 | abs(cases$lo/cases$n-.01)<1e-12)) |
  (cases$p==100 & cases$family=="graph" & (cases$path>0 | abs(cases$lo/cases$n-.1)<1e-12)),]

write.table(cases,file.path(out,"cases.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
for (seed in as.integer(args[-1])) {
  set.seed(seed); d <- file.path(out,seed); dir.create(d,showWarnings=FALSE)
  for (n in c(1000,10000,100000)) {
    t <- seq(0,1,length.out=n)
    y <- sin(4*pi*t)+.6*(t>.55)+rnorm(n,sd=.2)
    put(y,file.path(d,sprintf("signal-%d-y.tsv",n)))
    if(n==10000) put(cumsum(runif(n,.8,1.2)),file.path(d,"irregular-x.tsv"))
  }
  for (p in c(40,100)) {
    n <- if(p==40) 200 else 1000
    X <- matrix(rnorm(n*p),n,p)
    # The larger design is deliberately more correlated.
    corr <- if(p==40) .5 else .95
    for(j in 2:p) X[,j] <- corr*X[,j-1]+sqrt(1-corr^2)*X[,j]
    b <- rep(c(0,.5,-.5),length.out=p) # many changes, not an easy single-block fit
    y <- as.vector(X%*%b)+rnorm(n,sd=.3)
    put(X,file.path(d,sprintf("reg-%d-X.tsv",p))); put(y,file.path(d,sprintf("reg-%d-y.tsv",p)))
  }
}
