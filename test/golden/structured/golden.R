# Independent reference fixtures. Run only under Slurm on Discovery.
library(genlasso)
stopifnot(as.character(packageVersion("genlasso")) == "1.6.1")
stopifnot(as.character(packageVersion("glmgen")) == "0.0.3")
root <- commandArgs(TRUE)[1]
dir.create(root, recursive=TRUE, showWarnings=FALSE)
put <- function(x, path) write.table(x, path, sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)
for (seed in 8101:8103) {
  set.seed(seed)
  d <- file.path(root, as.character(seed)); dir.create(d, showWarnings=FALSE)
  n <- 40; p <- 8
  X <- matrix(rnorm(n*p), n, p)
  beta <- c(0,0,1,1,1,-0.5,-0.5,0)
  y <- as.vector(X %*% beta) + rnorm(n, sd=.3)
  D <- diff(diag(p))
  G <- rbind(D, c(1,0,-1,0,0,0,0,0), c(0,0,0,0,1,0,0,-1))
  F <- rbind(.25*diag(p), .75*D)
  levels <- c(.4,.12,.03,.006)
  put(X,file.path(d,"X.tsv")); put(y,file.path(d,"y.tsv"))
  put(G,file.path(d,"graph_D.tsv")); put(levels,file.path(d,"levels.tsv"))
  for (family in c("graph","fused")) {
    penalty <- if (family=="graph") G else F
    path <- genlasso(y, X=X, D=penalty, minlam=0, maxsteps=5000, svd=TRUE)
    stopifnot(path$completepath)
    z <- coef(path, lambda=n*levels)
    put(z$beta,file.path(d,paste0(family,"_beta.tsv")))
    put(z$df,file.path(d,paste0(family,"_df.tsv")))
  }
  # Both unit and irregular grids, with the same data for both R references.
  m <- 32
  ys <- sin((1:m)/5) + .35*rnorm(m)
  put(ys,file.path(d,"signal.tsv"))
  for (grid in c("unit","irregular")) {
    xs <- if (grid=="unit") 1:m else cumsum(runif(m,.8,1.2))
    put(xs,file.path(d,paste0(grid,"_x.tsv")))
    for (k in 0:3) {
      tag <- paste0(grid,"_k",k)
      a <- genlasso::trendfilter(ys,pos=xs,ord=k,minlam=0,maxsteps=10000,rtol=1e-9,btol=1e-9)
      stopifnot(a$completepath)
      z <- coef(a,lambda=m*levels)
      put(z$beta,file.path(d,paste0(tag,"_genlasso.tsv")))
      put(z$df,file.path(d,paste0(tag,"_df.tsv")))
      b <- glmgen::trendfilter(xs,ys,k=k,lambda=m*levels,thinning=FALSE,
                 control=glmgen::trendfilter.control.list(obj_tol=1e-12,max_iter=20000L))
      put(b$beta,file.path(d,paste0(tag,"_glmgen.tsv")))
      put(b$status,file.path(d,paste0(tag,"_status.tsv")))
    }
  }
}
writeLines(c(capture.output(sessionInfo()),
 "glmgen commit 9e6061503d6dd103712a0c07f6be7293c1c58aaf; lambda_R = n*lambda_Julia"),
 file.path(root,"provenance.txt"))
