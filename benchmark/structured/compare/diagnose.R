library(genlasso)
input <- commandArgs(TRUE)[1]; out <- commandArgs(TRUE)[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
seed <- 8203; p <- 40; n <- 200
X <- as.matrix(read.table(file.path(input,seed,"reg-40-X.tsv")))
y <- as.vector(as.matrix(read.table(file.path(input,seed,"reg-40-y.tsv"))))
DD <- diff(diag(p)); extra <- matrix(0,p%/%5,p)
for(j in 1:nrow(extra)) {extra[j,j]<- -1; extra[j,p-j+1]<-1}
D <- rbind(DD,extra); levels <- exp(seq(log(n*.1),log(n*.003),length.out=50))
jb <- matrix(readBin(file.path("baseline-canary-v2","julia-8203-graph-path-p40.bin"),"double",n=p*50),p,50)
obj <- function(b) colSums((X%*%b-y)^2)/(2*n)+levels/n*colSums(abs(D%*%b))
for (tol in c(1e-7,1e-9,1e-11)) for (complete in c(FALSE,TRUE)) {
  z <- genlasso(y,X=X,D=D,minlam=if(complete) 0 else min(levels),maxsteps=10000,svd=TRUE,rtol=tol,btol=tol)
  both <- coef(z,lambda=levels,type="both"); b <- both$beta; u <- both$u
  tag <- sprintf("graph-8203-tol%g-complete%s",tol,complete)
  writeBin(as.double(b),file.path(out,paste0(tag,".bin")))
  writeLines(c(sprintf("minlambda=%g complete=%s knots=%d",min(z$lambda),z$completepath,length(z$lambda)),
    paste(obj(b),collapse="\t")),file.path(out,paste0(tag,".txt")))
  errors <- sqrt(colSums((b-jb)^2))/sqrt(colSums(jb^2))
  kkt <- sqrt(colSums((crossprod(X,X%*%b-y)+crossprod(D,u))^2))
  write.table(data.frame(lambda=levels,error=errors,objR=obj(b),objJulia=obj(jb),stationarity=kkt),
    file.path(out,paste0(tag,".tsv")),sep="\t",row.names=FALSE,quote=FALSE)
  if(tol==1e-7 && !complete) {
    j <- which.max(errors); lam <- levels[j]
    H <- crossprod(X); Q <- D%*%solve(H,t(D)); d <- as.vector(D%*%solve(H,crossprod(X,y)))
    opt <- optim(pmin(lam,pmax(-lam,u[,j])),function(v) sum(v*(Q%*%v))/2-sum(d*v),
      function(v)as.vector(Q%*%v-d),method="L-BFGS-B",lower=-lam,upper=lam,
      control=list(factr=1e3,pgtol=1e-10,maxit=10000))
    bopt <- solve(H,crossprod(X,y)-crossprod(D,opt$par))
    grad <- as.vector(Q%*%opt$par-d)
    pg <- max(abs(opt$par-pmin(lam,pmax(-lam,opt$par-grad))))
    writeLines(c(sprintf("lambda=%g column=%d optim_status=%d projected_gradient=%g",lam,j,opt$convergence,pg),
      sprintf("relative_error_vs_Julia=%g relative_error_vs_genlasso=%g",sqrt(sum((bopt-jb[,j])^2)/sum(jb[,j]^2)),
        sqrt(sum((bopt-b[,j])^2)/sum(b[,j]^2)))),file.path(out,"independent-dual-QP.txt"))
  }
}
