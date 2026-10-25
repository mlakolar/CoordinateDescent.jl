# Independent bounded dual QP for the correlated sparse-fusion example.
# Its feasible dual objective supplies a lower bound even before optimization
# terminates. This check does not call either package's fitting routine.
out <- commandArgs(TRUE)[1]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
p <- 100L; n <- 1000L; lambda <- 3
D <- rbind(.25*diag(p),.75*diff(diag(p)))
rows <- list()
for(seed in 8201:8203) {
  X <- as.matrix(read.table(file.path("inputs",seed,"reg-100-X.tsv")))
  y <- as.vector(as.matrix(read.table(file.path("inputs",seed,"reg-100-y.tsv"))))
  H <- crossprod(X); h <- crossprod(X,y)
  Q <- D%*%solve(H,t(D)); d <- as.vector(D%*%solve(H,h))
  z <- optim(rep(0,nrow(D)),function(u)sum(u*(Q%*%u))/2-sum(d*u),
    function(u)as.vector(Q%*%u-d),method="L-BFGS-B",lower=-lambda,upper=lambda,
    control=list(factr=100,pgtol=1e-10,maxit=20000))
  u <- pmin(lambda,pmax(-lambda,z$par)); r <- h-crossprod(D,u)
  dual <- as.numeric(sum(y^2)/2-crossprod(r,solve(H,r))/2)
  gradient <- as.vector(Q%*%u-d)
  projected <- max(abs(u-pmin(lambda,pmax(-lambda,u-gradient))))
  for(solver in c("julia","genlasso-special","genlasso-qr","genlasso")) {
    folder <- switch(solver,julia="optimized-gram-matched",`genlasso-special`="reference-special",
      `genlasso-qr`="reference-qr",genlasso="baseline-full")
    beta <- matrix(readBin(file.path(folder,sprintf("%s-%d-fused-path-p100.bin",solver,seed)),"double",n=p*50),p,50)[,50]
    primal <- sum((y-X%*%beta)^2)/2+lambda*sum(abs(D%*%beta))
    rows[[length(rows)+1]] <- data.frame(seed,solver,lambda_R=lambda,primal=primal/n,dual=dual/n,
      relative_duality_gap=(primal-dual)/abs(primal),projected_dual_gradient=projected,
      max_dual_constraint_violation=max(0,max(abs(u))-lambda),optimizer_status=z$convergence)
  }
}
write.table(do.call(rbind,rows),file.path(out,"dual-check.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
