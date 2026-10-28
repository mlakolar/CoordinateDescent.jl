# Run with R 4.5.3 from a directory that holds references/ (see README.md).
# All output paths are run-local.
args <- commandArgs(TRUE)
out <- args[1]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
libs <- c(file.path(getwd(),"Rlib"),.libPaths())
dir.create(libs[1],showWarnings=FALSE)
.libPaths(libs)
if(!requireNamespace("linprog",quietly=TRUE))
  install.packages("references/linprog-f05bd8a.tar.gz",lib=libs[1],repos=NULL,type="source")
if(!requireNamespace("hdi",quietly=TRUE))
  install.packages("references/hdi-3eb71fb.tar.gz",lib=libs[1],repos=NULL,type="source")
library(hdi); library(glmnet)
writeLines(c(capture.output(sessionInfo()),paste("hdi",packageVersion("hdi"))),file.path(out,"R-session.txt"))
set.seed(77077)
n <- 80; p <- 12
X <- matrix(rnorm(n*p),n,p)
for(j in 2:p) X[,j] <- .4*X[,j-1]+sqrt(1-.4^2)*X[,j]
X <- scale(X,center=TRUE,scale=FALSE)
X <- scale(X,center=FALSE,scale=sqrt(colMeans(X^2)))
y <- as.vector(1+X[,1]-.8*X[,3]+rnorm(n))
lambda <- .08; node.lambda <- .1
beta <- as.vector(coef(glmnet(X,y,lambda=lambda,standardize=FALSE,intercept=TRUE,thresh=1e-14,maxit=1000000)))[-1]
Z <- sapply(1:p,function(j) {
  f <- glmnet(X[,-j],X[,j],lambda=node.lambda,standardize=FALSE,intercept=FALSE,thresh=1e-14,maxit=1000000)
  X[,j]-as.vector(predict(f,X[,-j],s=node.lambda))
})
R <- hdi::lasso.proj(X,y,standardize=FALSE,betainit=beta,sigma=1,Z=Z,return.Z=TRUE,suppress.grouptesting=TRUE)
write.csv(X,file.path(out,"X.csv"),row.names=FALSE)
write.csv(data.frame(y=y),file.path(out,"y.csv"),row.names=FALSE)
write.csv(data.frame(beta=beta,bhat=R$bhat,se=R$se,pvalue=R$pval),file.path(out,"hdi.csv"),row.names=FALSE)
write.csv(R$Z,file.path(out,"Z-raw.csv"),row.names=FALSE)
# return.Z returns the raw residuals; hdi normalizes them internally for inference.
write.csv(scale(R$Z,center=FALSE,scale=colMeans(R$Z*X)),file.path(out,"Z.csv"),row.names=FALSE)

# Preserve the full hdi bootstrap code path while fixing nuisance fitting and
# random errors: this removes unrelated RNG and CV differences across languages.
# hdi internally standardizes/centers X and y before calling these helpers.
ns <- asNamespace("hdi")
writeLines(deparse(get("boot.initial.fit",ns)),file.path(out,"hdi-boot.initial.fit.txt"))
writeLines(deparse(get("initial.estimator",ns)),file.path(out,"hdi-initial.estimator.txt"))
writeLines(deparse(get("boot.se",ns)),file.path(out,"hdi-boot.se.txt"))
writeLines(deparse(get("boot.lasso.proj",ns)),file.path(out,"hdi-boot.lasso.proj.txt"))
B <- 199
rc <- as.vector(y-mean(y)-X%*%beta); rc <- rc-mean(rc)
set.seed(77078)
E <- matrix(sample(rc,n*B,replace=TRUE),n,B)
write.csv(E,file.path(out,"errors.csv"),row.names=FALSE)
replacement <- function(betainit,sigma,x,y) list(beta.lasso=beta,sigmahat=1,lambda=lambda)
unlockBinding("initial.estimator",ns); assign("initial.estimator",replacement,ns); lockBinding("initial.estimator",ns)
replacement.resample <- function(r,B,wild) E
unlockBinding("resample",ns); assign("resample",replacement.resample,ns); lockBinding("resample",ns)
# Fixed exact lambda avoids glmnet's default interpolation between path knots.
replacement.fit <- function(x,y,initial.lasso.method,lambda,verbose=FALSE) {
  f <- glmnet(x,y,lambda=lambda,standardize=FALSE,intercept=TRUE,thresh=1e-14,maxit=1000000)
  cf <- as.vector(coef(f)); residual <- as.vector(y-predict(f,x))
  list(betalasso=cf[-1],intercept=cf[1],lambda=lambda,
       sigmahat=sqrt(sum(residual^2)/(nrow(x)-sum(cf!=0))))
}
unlockBinding("do.initial.fit",ns); assign("do.initial.fit",replacement.fit,ns); lockBinding("do.initial.fit",ns)
BR <- hdi::boot.lasso.proj(X,y,standardize=FALSE,betainit="cv lasso",sigma=1,Z=Z,return.bootdist=TRUE,B=B,boot.shortcut=TRUE)
write.csv(data.frame(bhat=BR$bhat,se=BR$se,pvalue=BR$pval,adjusted=BR$pval.corr),file.path(out,"hdi-bootstrap.csv"),row.names=FALSE)
write.csv(BR$cboot.dist,file.path(out,"hdi-bootstrap-draws.csv"),row.names=FALSE)
write.csv(BR$cboot.dist.underH0c,file.path(out,"hdi-bootstrap-null-draws.csv"),row.names=FALSE)

# Authors' unmodified QP routine, loaded without package-loading statements.
src <- readLines("references/sslasso_code/lasso_inference.r")
eval(parse(text=src[!grepl("^library",src)]))
mu <- .12
G <- crossprod(X)/n
JM <- t(sapply(1:p,function(j) InverseLinftyOneRow(G,j,mu,maxiter=100000,threshold=1e-10)$optsol))
write.csv(JM,file.path(out,"jm-theta.csv"),row.names=FALSE)

# Independent low-dimensional GLM checks of both the MLE limit and null score.
set.seed(77079)
yb <- rbinom(n,1,plogis(.2+.3*X[,1]))
yp <- rpois(n,exp(.2+.1*X[,1]))
write.csv(data.frame(binomial=yb,poisson=yp),file.path(out,"glm-y.csv"),row.names=FALSE)
for(fam in c("binomial","poisson")) {
  yy <- if(fam=="binomial") yb else yp
  F <- get(fam,mode="function")()
  fit <- glm(yy~X,family=F,control=glm.control(epsilon=1e-12,maxit=100))
  write.csv(data.frame(beta=coef(fit)[-1],se=sqrt(diag(vcov(fit)))[-1]),file.path(out,paste0(fam,"-mle.csv")),row.names=FALSE)
  j <- 2; null <- .25; N <- X[,-j]
  restricted <- glm(yy~N+offset(null*X[,j]),family=F,control=glm.control(epsilon=1e-12,maxit=100))
  muhat <- fitted(restricted)
  h <- if(fam=="binomial") muhat*(1-muhat) else muhat
  z <- residuals(lm(X[,j]~N,weights=h))
  info <- sum(h*z*X[,j])/n
  score <- sum(z*(muhat-yy))/n
  write.csv(data.frame(score=score,information=info,statistic=sqrt(n)*score/sqrt(info)),file.path(out,paste0(fam,"-score.csv")),row.names=FALSE)
}
writeLines("REFERENCE_COMPLETE",file.path(out,"COMPLETE"))
