# R references are generated on Discovery; shared X, normal draws and pairs
# remove cross-language RNG differences. No performance measurements here.
args <- commandArgs(TRUE); out <- args[1]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(Sys.getenv('SELECTION_RLIB'),.libPaths()))
write <- function(x,name) write.table(as.matrix(x),file.path(out,paste0(name,'.csv')),sep=',',row.names=FALSE,col.names=FALSE,quote=FALSE)
set.seed(8203)
n <- 90; p <- 12
Sigma <- outer(1:p,1:p,function(i,j) 0.4^abs(i-j))
mu <- seq(-0.2,0.3,length.out=p)
X <- sweep(matrix(rnorm(n*p),n,p)%*%chol(Sigma),2,mu,'+')
y <- 1 + X%*%c(1.5,-1,0.8,rep(0,p-3)) + rnorm(n)
lambda <- qnorm(1 - 0.1*(1:p)/(2*p))/sqrt(n)
factors <- c(3,2,1,0.6)
for(intercept in c(FALSE,TRUE)) for(std in c(FALSE,TRUE)) {
  fit <- SLOPE::SLOPE(X,y,intercept=intercept,center=if(intercept) 'mean' else 'none',
    scale=if(std) 'sd' else 'none',alpha=factors,lambda=lambda,gamma=0,
    tol=1e-10,max_passes=100000,threads=1,tol_dev_ratio=1,max_variables=p+1)
  write(coef(fit,intercept=intercept),paste0('slope-',as.integer(intercept),'-',as.integer(std)))
}
s <- knockoff::create.solve_equi(Sigma)
set.seed(8204); Z <- matrix(rnorm(n*p),n,p)
set.seed(8204); K <- knockoff::create.gaussian(X,mu,Sigma,method='equi',diag_s=s)
write(s,'s');write(Z,'Z');write(K,'Xk')
# A shared augmented design compares the actual coefficient-difference statistic.
fit <- glmnet::glmnet(cbind(X,K),y,lambda=0.12,intercept=FALSE,standardize=FALSE,thresh=1e-14,maxit=1000000)
b <- as.numeric(coef(fit))[-1]
W <- abs(b[1:p])-abs(b[p+(1:p)])
write(W,'W')
write(knockoff::knockoff.threshold(W,fdr=0.3,offset=1),'threshold')
# Basis-independent fixed-X agreement. R normalizes columns to l2=1.
Xfixed <- sweep(X,2,sqrt(colSums(X^2)),'/')
Kfixed <- knockoff::create.fixed(Xfixed,method='equi')$Xk
write(Xfixed,'Xfixed');write(crossprod(Kfixed),'fixed-gram');write(crossprod(Xfixed,Kfixed),'fixed-cross')
# stabs' fixed-lambda max-coefficient selector is the same cap when at least
# q coefficients are nonzero. Use no standardization and identical halves.
B <- 10; cap <- 4; h <- n%/%2
folds <- matrix(0,n,2*B)
for(k in 1:B) { rows <- sample.int(n); folds[rows[1:h],2*k-1] <- 1; folds[rows[(h+1):(2*h)],2*k] <- 1 }
fitfun <- function(x,y,q,...) {
  f <- glmnet::glmnet(x,y,lambda=0.005,intercept=FALSE,standardize=FALSE,thresh=1e-14,maxit=1000000)
  b <- abs(as.numeric(coef(f))[-1]); selected <- rep(FALSE,ncol(x))
  selected[head(order(b,decreasing=TRUE),min(q,sum(b>0)))] <- TRUE
  list(selected=selected)
}
st <- stabs::stabsel(X,y,fitfun=fitfun,cutoff=0.8,q=cap,folds=folds,B=B,
  assumption='none',sampling.type='SS',papply=lapply,verbose=FALSE)
write(folds,'folds');write(st$max,'stability-frequency');write(st$selected,'stability-selected')
write(st$PFER,'pfer')
write(X,'X');write(y,'y');write(Sigma,'Sigma');write(mu,'mu');write(lambda,'lambda');write(factors,'factors')
writeLines(capture.output(sessionInfo()),file.path(out,'R-session.txt'))
cat('GOLDEN_GENERATION_DONE\n')
