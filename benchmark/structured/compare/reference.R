# One serial R worker per independent dataset. All execution is under Slurm.
library(genlasso)
args <- commandArgs(TRUE)
seed <- as.integer(args[1]); input <- args[2]; out <- args[3]; mode <- args[4]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
cases <- read.delim(file.path(input,"cases.tsv"),check.names=FALSE)
if(mode=="canary") cases <- cases[cases$id %in% c("signal-n1000-k2-l1","signal-n10000-k2-l10","graph-path-p40"),]
if(mode %in% c("special","qr")) cases <- cases[cases$family!="signal",]
if(mode=="loose") cases <- cases[cases$family=="signal",]
read <- function(name) as.matrix(read.table(file.path(input,seed,name),header=FALSE))
rss <- function() {
  z <- readLines("/proc/self/status"); as.numeric(strsplit(trimws(sub("VmHWM:","",z[grepl("^VmHWM:",z)]))," +")[[1]][1])*1024
}
objective <- function(B,y,X,D,levels) {
  if(is.null(dim(B))) B <- matrix(B,ncol=1)
  fit <- if(is.null(X)) B else X%*%B
  penalty <- if(is.null(D)) apply(B,2,function(b)sum(abs(diff(b)))) else colSums(abs(D%*%B))
  colSums((fit-y)^2)/(2*length(y))+levels*penalty/length(y)
}
con <- file(file.path(out,paste0("R-",seed,"-",mode,".tsv")),"wt")
writeLines("seed\tcase\tsolver\tround\tseconds\tconverged\titerations\tobjective\tworker_hwm_bytes\tbatch\tstatus",con)
# Compile/load native routines on unrelated data before any measured calls.
invisible(glmgen::trendfilter(sin(1:32),k=2,lambda=1,thinning=FALSE))
invisible(genlasso::trendfilter(sin(1:32),ord=2,minlam=.1))
for (ii in seq_len(nrow(cases))) {
  c <- cases[ii,]; X <- D <- NULL
  levels <- if(c$path>0) exp(seq(log(c$hi),log(c$lo),length.out=c$path)) else c$lo
  if(c$family=="signal") {
    y <- as.vector(read(sprintf("signal-%d-y.tsv",c$n)))
    x <- if(c$grid=="unit") seq_along(y) else as.vector(read("irregular-x.tsv"))
    solvers <- if(mode=="loose") c("glmgen-default","glmgen-1e8") else if(mode=="million")
      as.character(c$r_setting) else c("glmgen",if(c$genlasso) "genlasso")
  } else {
    X <- read(sprintf("reg-%d-X.tsv",c$p)); y <- as.vector(read(sprintf("reg-%d-y.tsv",c$p)))
    DD <- diff(diag(c$p))
    D <- if(c$family=="fused") rbind(.25*diag(c$p),.75*DD) else if(c$family=="trend") diff(DD) else {
      # Chain plus long edges produces cycles, unlike the first timing campaign.
      extra <- matrix(0,c$p%/%5,c$p)
      for(j in 1:nrow(extra)) { extra[j,j] <- -1; extra[j,c$p-j+1] <- 1 }
      rbind(DD,extra)
    }
    solvers <- if(mode=="special") "genlasso-special" else if(mode=="qr") "genlasso-qr" else "genlasso"
  }
  for(solver in solvers) {
    fit <- function(tol=if(solver=="glmgen-default") 1e-5 else if(solver=="glmgen-1e8") 1e-8 else 1e-10) {
      if(startsWith(solver,"glmgen")) {
        z <- glmgen::trendfilter(x,y,k=c$order,lambda=levels,thinning=FALSE,
          control=glmgen::trendfilter.control.list(obj_tol=tol,max_iter=20000L))
        list(B=z$beta,flag=all(z$status==0),it=sum(z$iter))
      } else if(solver=="genlasso-special") {
        scale <- if(c$family=="fused") .75 else 1
        z <- if(c$family=="trend") genlasso::trendfilter(y,X=X,ord=1,minlam=min(levels),maxsteps=10000) else
          genlasso::fusedlasso(y,X=X,D=if(c$family=="fused") DD else D,
            gamma=if(c$family=="fused") 1/3 else 0,minlam=scale*min(levels),maxsteps=10000)
        list(B=coef(z,lambda=scale*levels)$beta,
             flag=min(z$lambda)<=scale*min(levels)*(1+1e-8) || isTRUE(z$completepath),it=length(z$lambda))
      } else {
        z <- if(c$family=="signal") genlasso::trendfilter(y,pos=x,ord=c$order,minlam=min(levels),
                 maxsteps=10000,rtol=1e-9,btol=1e-9) else
                 genlasso::genlasso(y,X=X,D=D,minlam=min(levels),maxsteps=10000,svd=solver!="genlasso-qr")
        list(B=coef(z,lambda=levels)$beta,
             flag=min(z$lambda)<=min(levels)*(1+1e-8) || isTRUE(z$completepath),it=length(z$lambda))
      }
    }
    # Independent, tighter reference for large signal comparisons. Reference
    # fitting is excluded from measured time and retained even if it fails.
    if(solver=="glmgen" || mode=="million") {
      strict <- tryCatch(fit(1e-12),error=function(e)NULL)
      if(!is.null(strict)) {
        f <- file(file.path(out,sprintf("tight-%d-%s.bin",seed,c$id)),"wb")
        writeBin(as.double(strict$B),f,size=8); close(f)
        writeLines(paste(strict$flag,strict$it),file.path(out,sprintf("tight-%d-%s.status",seed,c$id)))
      }
    }
    for(round in 1:3) {
    gc(); result <- tryCatch({
      setTimeLimit(elapsed=if(mode=="million") 600 else 180,transient=TRUE)
      elapsed <- system.time({ z <- fit() })[["elapsed"]]
      batch <- 1L
      if(elapsed<.02) {
        batch <- min(200L,ceiling(.05/max(elapsed,.0001)))
        elapsed <- system.time(for(j in seq_len(batch)) z <- fit())[["elapsed"]]/batch
      }
      B <- z$B; flag <- z$flag; it <- z$it
      setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
      if(round==1) {
        f <- file(file.path(out,sprintf("%s-%d-%s.bin",solver,seed,c$id)),"wb")
        writeBin(as.double(B),f,size=8); close(f)
      }
      # Store objective at the least regularized level; complete accuracy checks
      # are performed identically in Julia, outside either solver's timing.
      ob <- if(c$family=="signal") NA_real_ else tail(objective(B,y,X,D,levels),1)
      c(elapsed,flag,it,ob,rss(),batch,"ok")
    },error=function(e) {
      setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
      c(NA,FALSE,NA,NA,rss(),NA,gsub("[\t\n]"," ",conditionMessage(e)))
    })
    writeLines(paste(c(seed,c$id,solver,round,result),collapse="\t"),con); flush(con)
    }
  }
}
close(con)
writeLines(c(capture.output(sessionInfo()),"glmgen: 9e6061503d6dd103712a0c07f6be7293c1c58aaf"),file.path(out,paste0("R-provenance-",seed,".txt")))
