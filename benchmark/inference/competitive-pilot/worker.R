# One independent R process per fit, under Slurm.
args <- commandArgs(TRUE)
engine <- args[1]; mode <- args[2]; scenario <- args[3]; rep <- as.integer(args[4])
libs <- c(file.path(getwd(),"Rlib"),"/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/Rlib",
          "/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3","/home1/mkolar/juliaPkg/Rlib-r4.5.3",.libPaths())
.libPaths(libs)
suppressPackageStartupMessages(library(glmnet))
dest <- file.path("data",scenario,rep)
out <- file.path("outputs",paste(engine,mode,scenario,rep,sep="-"))
config <- read.csv(file.path(dest,"config.csv"))
X <- as.matrix(read.csv(file.path(dest,"X.csv"),header=FALSE))
y <- scan(file.path(dest,"y.csv"),quiet=TRUE)
targets <- scan(file.path(dest,"targets.csv"),quiet=TRUE)
family <- config$family
lambda <- config$lambda_main; node <- config$lambda_node
if(engine=="hdi") suppressPackageStartupMessages(library(hdi))
direction.calls <- list()
if(engine=="SIHR") {
  suppressPackageStartupMessages(library(SIHR))
  ns <- asNamespace("SIHR")
  original.direction <- get("Direction_fixedtuning",ns)
  tracked.direction <- function(...) {
    ans <- original.direction(...)
    direction.calls[[length(direction.calls)+1L]] <<- list(mu=ans$mu,status=ans$status)
    ans
  }
  unlockBinding("Direction_fixedtuning",ns)
  assign("Direction_fixedtuning",tracked.direction,ns)
  lockBinding("Direction_fixedtuning",ns)
}
author <- new.env()
if(engine=="sslasso") {
  suppressPackageStartupMessages(library(flare))
  src <- readLines("/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/references/sslasso_code/lasso_inference.r")
  if(mode=="matched") {
    # In the diagonal fast path, the exact optimum is (1-mu)/G_ii.
    # The original uses mu0 (the threshold for entering the branch), yielding
    # a feasible but nonoptimal row whenever mu > mu0. Change only this return
    # branch; retain the iterative solver initialization and all other code.
    at <- grep("if (mu >= mu0)",src,fixed=TRUE)
    stopifnot(length(at)==1L,grepl("(1-mu0)",src[at+1L],fixed=TRUE))
    src[at+1L] <- sub("(1-mu0)","(1-mu)",src[at+1L],fixed=TRUE)
  }
  eval(parse(text=src[!grepl("^library",src)]),envir=author)
  original.inverse <- author$InverseLinfty
  author$InverseLinfty <- function(...) {
    M <- original.inverse(...)
    author$last_M <- M
    M
  }
  if(mode=="matched") {
    # Authors' explicit-lambda branch references the nonexistent variable Las.
    # Replace only this fitting helper with an exact-lambda glmnet fit. Keep
    # their QP, noise estimator, CI inflation and p-value code unchanged.
    author$Lasso <- function(X,y,lambda=NULL,intercept=TRUE) {
      stopifnot(!is.null(lambda))
      g <- glmnet(X,y,lambda=lambda,intercept=intercept,thresh=1e-10,maxit=100000)
      beta <- as.vector(coef(g))
      if(intercept) beta else beta[-1]
    }
  }
}

fit <- function(A,b,ts) {
  p <- ncol(A); n <- nrow(A)
  set.seed(770000+rep)
  if(engine=="hdi") {
    if(mode=="default") {
      f <- hdi::lasso.proj(A,b,family=family)
    } else {
      # Public hdi GLM preprocessing internally estimates Fisher weights and
      # exposes no matching initial-penalty hook; do not silently relabel it.
      if(family!="gaussian") stop("UNSUPPORTED: native hdi GLM has no public matched-initial-Fisher hook")
      g <- glmnet(A,b,lambda=lambda,standardize=FALSE,intercept=TRUE,thresh=1e-10,maxit=100000)
      beta <- as.vector(coef(g))[-1]
      res <- b - as.vector(predict(g,A))
      df <- n-sum(beta!=0)-1
      if(df<=0) stop("no residual degrees of freedom")
      Z <- vapply(seq_len(p),function(j) {
        f <- glmnet(A[,-j,drop=FALSE],A[,j],lambda=node,standardize=FALSE,
                    intercept=FALSE,thresh=1e-10,maxit=100000)
        A[,j]-as.vector(predict(f,A[,-j,drop=FALSE]))
      },numeric(n))
      f <- hdi::lasso.proj(A,b,standardize=FALSE,betainit=beta,sigma=sqrt(sum(res^2)/df),
                           Z=Z,suppress.grouptesting=TRUE)
    }
    return(list(beta=f$bhat[ts],se=f$se[ts],low=f$bhat[ts]-qnorm(.975)*f$se[ts],
                high=f$bhat[ts]+qnorm(.975)*f$se[ts],pval=f$pval[ts],actual=p))
  }
  if(engine=="sslasso") {
    kw <- list(X=A,y=b,verbose=FALSE)
    if(mode=="matched") kw <- c(kw,list(lambda=lambda,mu=node,maxiter=100000,threshold=1e-7))
    f <- do.call(author$SSLasso,kw)
    return(list(beta=f$unb.coef[ts],se=rep(NA_real_,length(ts)),low=f$low.lim[ts],
                high=f$up.lim[ts],pval=f$pvals[ts],actual=p))
  }
  if(engine=="SIHR") {
    L <- matrix(0,p,length(ts)); L[cbind(ts,seq_along(ts))] <- 1
    kw <- list(X=A,y=b,loading.mat=L,model=if(family=="gaussian") "linear" else "logistic")
    if(mode=="matched") kw <- c(kw,list(lambda=lambda,mu=node,rescale=1,prob.filter=0))
    # Keep SIHR's glmnet initializer at the exact same nominal lambda. Its
    # projection objective differs from nodewise lasso; matching mu does not
    # make the inferential methods algebraically identical.
    f <- do.call(SIHR::LF,kw)
    return(list(beta=f$est.debias.vec,se=f$se.vec,low=f$est.debias.vec-qnorm(.975)*f$se.vec,
                high=f$est.debias.vec+qnorm(.975)*f$se.vec,
                pval=2*pnorm(-abs(f$est.debias.vec/f$se.vec)),actual=length(ts)))
  }
  stop("unknown engine")
}

warnings <- character()
result <- tryCatch(withCallingHandlers({
  if(engine=="hdi" && mode=="matched" && family!="gaussian")
    stop("UNSUPPORTED: native hdi GLM has no public matched-initial-Fisher hook")
  set.seed(177)
  A <- matrix(rnorm(80*12),80,12)
  A <- scale(A,scale=FALSE); A <- scale(A,center=FALSE,scale=sqrt(colMeans(A^2)))
  b <- if(family=="gaussian") rnorm(80) else rbinom(80,1,.5)
  fit(A,b,c(1,2))
  warnings <- character()
  direction.calls <- list()
  gc()
  t <- proc.time()[[3]]
  f <- fit(X,y,targets)
  elapsed <- proc.time()[[3]]-t
  qp.diag <- list()
  if(engine=="SIHR" && length(direction.calls)) {
    actual.mu <- vapply(direction.calls,function(x) x$mu,numeric(1))
    qp.diag <- list(projection_mu_min=min(actual.mu),projection_mu_max=max(actual.mu),
                    projection_solver_calls=length(direction.calls),
                    projection_solver_statuses=paste(unique(vapply(direction.calls,function(x) as.character(x$status),character(1))),collapse=" | "))
  }
  if(engine=="sslasso" && mode=="matched" && !is.null(author$last_M)) {
    # Post-timing numerical diagnostics, without changing author output.
    A <- cbind(1,scale(X,center=FALSE,scale=sqrt(colMeans(X^2))))
    G <- crossprod(A)/nrow(A)
    ids <- targets+1L
    M <- author$last_M[ids,,drop=FALSE]
    MG <- M %*% G
    E <- matrix(0,nrow(M),ncol(M)); E[cbind(seq_along(ids),ids)] <- 1
    gaps <- rowSums(MG*M)-M[cbind(seq_along(ids),ids)]+node*rowSums(abs(M))
    qp.diag <- list(qp_max_residual=max(abs(MG-E)),qp_max_abs_gap=max(abs(gaps)))
    # NoiseSd uses all debiased coordinates, so certify the full inverse too.
    M.all <- author$last_M
    MG.all <- M.all %*% G
    gaps.all <- rowSums(MG.all*M.all)-diag(M.all)+node*rowSums(abs(M.all))
    qp.diag <- c(qp.diag,list(qp_max_residual_all=max(abs(MG.all-diag(ncol(G)))),
                              qp_max_abs_gap_all=max(abs(gaps.all))))
  }
  ans <- data.frame(target=targets,estimate=as.vector(f$beta),se=as.vector(f$se),
                    lower=as.vector(f$low),upper=as.vector(f$high),pvalue=as.vector(f$pval))
  if(!all(is.finite(as.matrix(ans[,c("estimate","lower","upper","pvalue")])))) stop("nonfinite inference output")
  write.csv(ans,paste0(out,".csv"),row.names=FALSE)
  c(list(status="ok",fit_seconds=elapsed,actual_inferred_targets=f$actual),qp.diag)
},warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")}),
error=function(e) list(status=if(grepl("^UNSUPPORTED",conditionMessage(e))) "unsupported" else "error",error=conditionMessage(e)))
result$warnings <- paste(unique(warnings),collapse=" | ")
writeLines(paste(names(result),unlist(result),sep="="),paste0(out,".meta"))
cat(paste(names(result),unlist(result),sep="="),sep="\n")
