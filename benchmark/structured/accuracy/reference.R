# Untimed, independently implemented references on fresh shared inputs.
library(genlasso)
args <- commandArgs(TRUE)
seed <- as.integer(args[1]); input <- args[2]; out <- args[3]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
cases <- read.delim(file.path(input,"cases.tsv"),check.names=FALSE)
read <- function(name) as.matrix(read.table(file.path(input,seed,name),header=FALSE))
rows <- list()
for(ii in seq_len(nrow(cases))) {
  c <- cases[ii,]
  levels <- if(c$path>0) exp(seq(log(c$hi),log(c$lo),length.out=c$path)) else c$lo
  X <- D <- NULL
  if(c$family=="signal") {
    y <- as.vector(read(sprintf("signal-%d-y.tsv",c$n)))
    x <- if(c$grid=="unit") seq_along(y) else as.vector(read("irregular-x.tsv"))
    z <- glmgen::trendfilter(x,y,k=c$order,lambda=levels,thinning=FALSE,
      control=glmgen::trendfilter.control.list(obj_tol=1e-12,max_iter=20000L))
    B <- z$beta; success <- all(z$status==0); iterations <- sum(z$iter)
  } else {
    X <- read(sprintf("reg-%d-X.tsv",c$p)); y <- as.vector(read(sprintf("reg-%d-y.tsv",c$p)))
    DD <- diff(diag(c$p)); extra <- matrix(0,c$p%/%5,c$p)
    for(j in seq_len(nrow(extra))) {extra[j,j] <- -1;extra[j,c$p-j+1] <- 1}
    D <- if(c$family=="fused") rbind(.25*diag(c$p),.75*DD) else rbind(DD,extra)
    scale <- if(c$family=="fused") .75 else 1
    z <- genlasso::fusedlasso(y,X=X,D=if(c$family=="fused") DD else D,
      gamma=if(c$family=="fused") 1/3 else 0,minlam=scale*min(levels),maxsteps=10000)
    B <- coef(z,lambda=scale*levels)$beta
    success <- min(z$lambda)<=scale*min(levels)*(1+1e-8) || isTRUE(z$completepath)
    iterations <- length(z$lambda)
    # A clipped box-constrained dual QP gives an independent feasible lower
    # bound at the least penalized requested level, even before QP convergence.
    H <- crossprod(X); h <- crossprod(X,y); alpha <- min(levels)
    Q <- D%*%solve(H,t(D)); d <- as.vector(D%*%solve(H,h))
    q <- optim(rep(0,nrow(D)),function(u)sum(u*(Q%*%u))/2-sum(d*u),
      function(u)as.vector(Q%*%u-d),method="L-BFGS-B",lower=-alpha,upper=alpha,
      control=list(factr=100,pgtol=1e-10,maxit=20000))
    u <- pmin(alpha,pmax(-alpha,q$par))
    writeBin(as.double(u),file.path(out,sprintf("dual-%s.bin",c$id)),size=8)
  }
  writeBin(as.double(B),file.path(out,sprintf("reference-%s.bin",c$id)),size=8)
  rows[[length(rows)+1]] <- data.frame(seed,case=c$id,reference=if(c$family=="signal") "glmgen-1e12" else "genlasso-special",
    converged=success,iterations=iterations,columns=ncol(as.matrix(B)))
  write.table(do.call(rbind,rows),file.path(out,"status.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
}
writeLines(c(capture.output(sessionInfo()),"glmgen source: 9e6061503d6dd103712a0c07f6be7293c1c58aaf"),file.path(out,"provenance.txt"))
if(!all(vapply(rows,function(x)x$converged,logical(1)))) stop("reference did not converge")
