suppressPackageStartupMessages(library(rrpack))
args <- commandArgs(TRUE)
input <- args[1]; out <- args[2]
phase <- if(length(args)>=3) args[3] else "main"
component_only <- length(args)>=4 && args[4]=="component_only"
memory_solver <- if(length(args)>=5) args[5] else ""
dir.create(out, recursive=TRUE, showWarnings=FALSE)
dims <- scan(file.path(input,"dims.txt"),quiet=TRUE)
n <- dims[1]; p <- dims[2]; q <- dims[3]; r <- dims[4]
scalars <- scan(file.path(input,"scalars.txt"),quiet=TRUE)
readmat <- function(name,nr,nc) {
  con <- file(file.path(input,paste0(name,".bin")),"rb")
  on.exit(close(con))
  matrix(readBin(con,"double",n=nr*nc,size=8,endian="little"),nr,nc)
}
Xraw <- readmat("X",n,p);Yraw <- readmat("Y",n,q)
A0 <- readmat("A0",p,r);V0 <- readmat("V0",q,r)
layer <- readmat("layer",n,q)
u0 <- as.vector(readmat("u0",p,1));v0 <- as.vector(readmat("v0",q,1))
bits <- strsplit(basename(input),"-",fixed=TRUE)[[1]]
case <- bits[1];replication <- as.integer(bits[2])
fields <- strsplit("phase,language,case,rep,family,solver,mode,frac,round,seconds,iterations,objective,gap,residual,status,native_converged,rss_peak_kb,rss_baseline_kb,detail",",",fixed=TRUE)[[1]]
rss <- function() {
  s <- readLines("/proc/self/status")
  as.numeric(gsub("[^0-9]","",s[grepl("^VmHWM:",s)]))
}
normf <- function(A) sqrt(sum(A*A))
ln1 <- function(x,g,lambda) max(ifelse(abs(x)>1e-12,abs(g+lambda*sign(x)),pmax(0,abs(g)-lambda)))
certificate <- function(X,Y,B,family,frac,A=NULL,V=NULL) {
  if(family=="rank") {
    obj <- sum((Y-X%*%B)^2)/(2*n)
    return(c(obj,abs(obj-scalars[5])/max(1,abs(obj)),0))
  }
  S <- svd(B,nu=r,nv=r)
  if(family=="sparse") {
    if(is.null(A)) {
      A <- sweep(S$u[,1:r,drop=FALSE],2,S$d[1:r],"*")
      V <- S$v[,1:r,drop=FALSE]
    }
    E <- X%*%A%*%t(V)-Y
    G <- crossprod(X,E)%*%V/n
    lambda <- frac*scalars[2]
    an <- sqrt(rowSums(A*A))
    errs <- pmax(0,sqrt(rowSums(G*G))-lambda)
    nz <- an>1e-12
    errs[nz] <- sqrt(rowSums((G[nz,,drop=FALSE]+A[nz,,drop=FALSE]*(lambda/an[nz]))^2))
    GV <- crossprod(E,X)%*%A/n
    tangent <- GV-V%*%((crossprod(V,GV)+crossprod(GV,V))/2)
    res <- max(max(errs)/max(1,normf(crossprod(X,Y)%*%V/n)),
               normf(tangent)/max(1,normf(crossprod(Y,X)%*%A/n)))
    return(c(sum(E*E)/(2*n)+lambda*sum(an),NaN,res))
  }
  if(S$d[1]<1e-14) return(c(sum(Y*Y)/(2*n),NaN,0))
  u <- S$u[,1];v <- S$v[,1]*S$d[1]
  E <- X%*%tcrossprod(u,v)-Y;lambda <- frac*scalars[3]
  gu <- as.vector(crossprod(X,E)%*%v/n)
  gv <- as.vector(crossprod(E,X)%*%u/n)
  res <- max(ln1(u,gu,lambda*sum(abs(v)))/max(1,normf(crossprod(X,Y)%*%v/n)),
             ln1(v,gv,lambda*sum(abs(u)))/max(1,normf(crossprod(Y,X)%*%u/n)))
  c(sum(E*E)/(2*n)+lambda*sum(abs(u))*sum(abs(v)),NaN,res)
}
fit <- function(solver,frac) {
  X <- sweep(Xraw,2,colMeans(Xraw))
  Y <- sweep(Yraw,2,colMeans(Yraw))
  if(solver=="r_rrr") {
    z <- rrpack::rrr.fit(Y,X,nrank=r,coefSVD=FALSE)
    return(list(B=z$coef,ok=TRUE))
  }
  if(solver=="r_rrr_qr") {
    z <- rrpack::rrr(Y,X,penaltySVD="rank",df.type="naive",maxrank=r,
         modstr=list(lambda=r),control=list(qr.tol=max(n,p)*.Machine$double.eps,sv.tol=1e-12))
    return(list(B=coef(z),ok=TRUE))
  }
  if(solver=="r_srrr") {
    z <- rrpack::srrr(Y,X,nrank=r,A0=A0,V0=V0,
        modstr=list(lamA=n*frac*scalars[2],nlam=1,WA=rep(1,p)),
        control=list(epsilon=1e-8,maxit=1000,inner.eps=1e-9,inner.maxit=10000))
    return(list(B=coef(z),ok=NA,A=matrix(z$A.path[1,,],p,r),
                V=matrix(z$V.path[1,,],q,r)))
  }
  z <- rrpack:::surr_fit_Rcpp(layer,X,n*frac*scalars[3],u0,v0,
       rep(1,p),rep(1,q),X,
       rrpack:::rssvd.control(maxit=1000,epsilon=1e-8,innerEpsilon=1e-9,innerMaxit=10000))
  list(B=tcrossprod(z$U,z$V)*z$D,ok=z$converged)
}
fit <- compiler::cmpfun(fit)
guarded <- function(solver,frac,profile=FALSE) {
  job <- parallel::mcparallel({
    baseline <- rss()
    t <- Sys.time()
    if(profile) {
      Rprof(file.path(out,paste0("profile-",solver,".Rprof")),interval=0.001)
      calls <- 0
      repeat {
        z <- fit(solver,frac); calls <- calls+1
        if(as.numeric(difftime(Sys.time(),t,units="secs"))>.25) break
      }
      Rprof(NULL)
      capture.output(summaryRprof(file.path(out,paste0("profile-",solver,".Rprof"))),
                     file=file.path(out,paste0("profile-",solver,".txt")))
    } else z <- fit(solver,frac)
    elapsed <- as.numeric(difftime(Sys.time(),t,units="secs"))
    X <- sweep(Xraw,2,colMeans(Xraw));Y <- sweep(Yraw,2,colMeans(Yraw))
    family <- if(solver %in% c("r_rrr","r_rrr_qr")) "rank" else if(solver=="r_srrr") "sparse" else "component"
    cert <- certificate(X,if(family=="component") layer else Y,z$B,family,frac,z$A,z$V)
    list(B=z$B,A=z$A,V=z$V,seconds=elapsed,cert=cert,ok=z$ok,rss=rss(),baseline=baseline)
  },silent=TRUE)
  started <- proc.time()[3]
  repeat {
    z <- parallel::mccollect(job,wait=FALSE)
    if(!is.null(z)) return(z[[1]])
    if(proc.time()[3]-started>30) {
      tools::pskill(job$pid,signal=9)
      parallel::mccollect(job,wait=TRUE)
      return(list(timeout=TRUE))
    }
    Sys.sleep(.01)
  }
}
solvers <- if(component_only) "r_component" else if(case=="responses") c("r_rrr","r_rrr_qr","r_srrr") else c("r_rrr","r_rrr_qr","r_srrr","r_component")
if(nchar(memory_solver)) solvers <- memory_solver
for(s in solvers) invisible(fit(s,.1))
output <- file.path(out,if(nchar(memory_solver)) "memory.csv" else if(component_only) "r-component.csv" else "r.csv")
writeLines(paste(fields,collapse=","),output)
repeats <- if(phase=="pilot" || nchar(memory_solver)) 1 else 3
for(round_ in seq_len(repeats)) {
  order <- ((seq_along(solvers)+replication+round_-2)%%length(solvers))+1
  for(solver in solvers[order]) {
    family <- if(solver %in% c("r_rrr","r_rrr_qr")) "rank" else if(solver=="r_srrr") "sparse" else "component"
    for(frac in if(family=="rank") 0 else if(nchar(memory_solver)) .1 else c(.5,.1,.02)) {
      z <- guarded(solver,frac)
      row <- as.list(rep(NA,length(fields)));names(row) <- fields
      row[c("phase","language","case","rep","family","solver","mode","frac","round")] <-
         list(phase,"r",case,replication,family,solver,if(nchar(memory_solver)) "memory" else "fit",frac,round_)
      if(inherits(z,"try-error")) {
        row$status <- "error";row$detail <- gsub("[,\n]",";",as.character(z))
      } else if(isTRUE(z$timeout)) {
        row$status <- "timeout";row$seconds <- 30;row$detail <- "owned child killed after30s"
      } else {
        row$seconds <- z$seconds;row$iterations <- -1
        row$objective <- z$cert[1];row$gap <- z$cert[2];row$residual <- z$cert[3]
        row$status <- if(z$cert[3]<=1e-6 && (is.na(z$cert[2])||z$cert[2]<=1e-6)) "passed" else "accuracy_failed"
        row$native_converged <- z$ok;row$rss_peak_kb <- z$rss;row$rss_baseline_kb <- z$baseline
        if(round_==1 && !nchar(memory_solver)) {
          con <- file(file.path(out,paste0(solver,"-",frac,".bin")),"wb")
          writeBin(as.double(z$B),con,size=8,endian="little");close(con)
          if(family=="sparse") for(tag in c("A","V")) {
            con <- file(file.path(out,paste0(solver,"-",frac,".",tag)),"wb")
            writeBin(as.double(z[[tag]]),con,size=8,endian="little");close(con)
          }
        }
      }
      write(paste(unlist(row),collapse=","),file=output,append=TRUE)
      cat(case,replication,solver,frac,row$status,"\n");flush.console()
    }
  }
}
if(phase!="pilot" && !nchar(memory_solver) && case %in% c("medium","responses","wide"))
  for(s in solvers) try(guarded(s,.1,profile=TRUE),silent=TRUE)
writeLines(c(paste("R",getRversion()),paste("rrpack",packageVersion("rrpack")),
             capture.output(extSoftVersion())),file.path(out,"r-environment.txt"))
