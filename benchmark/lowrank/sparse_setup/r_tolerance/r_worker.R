suppressPackageStartupMessages(library(rrpack))
args <- commandArgs(TRUE); input <- args[1]; out <- args[2]; mode <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
dims <- scan(file.path(input,"dims.txt"),quiet=TRUE)
n <- dims[1]; p <- dims[2]; q <- dims[3]; r <- 3
scalars <- scan(file.path(input,"scalars.txt"),quiet=TRUE)
readmat <- function(key,nr,nc=1) {
  con <- file(file.path(input,paste0(key,".bin")),"rb");on.exit(close(con))
  matrix(readBin(con,"double",n=nr*nc,size=8,endian="little"),nr,nc)
}
savebin <- function(x,path) {con <- file(path,"wb");writeBin(as.double(x),con,size=8,endian="little");close(con)}
X <- readmat("X",n,p);Y <- readmat("Y",n,q)
Xc <- sweep(X,2,colMeans(X));Yc <- sweep(Y,2,colMeans(Y))
A0 <- readmat("A0",p,r);V0 <- readmat("V0",q,r);adaptive <- as.vector(readmat("adaptive",p))
bits <- strsplit(basename(input),"-",fixed=TRUE)[[1]];case <- bits[1];replication <- as.integer(bits[2])
configs <- data.frame(frac=c(.5,.1,.02),adaptive=FALSE)
if(case=="medium") configs <- rbind(configs,data.frame(frac=.1,adaptive=TRUE))
tag <- function(c) paste("r",c$frac,if(c$adaptive) "adaptive" else "unit",sep="-")
fit <- function(c) {
  w <- if(c$adaptive) adaptive else rep(1,p)
  lam <- c$frac*scalars[if(c$adaptive) 2 else 1]
  if(mode=="common") {
    z <- rrpack:::srrr_Rcpp(Yc,Xc,"glasso",A0,V0,r,n*lam,1e-8,1000L,1e-9,10000L,w)
    return(list(B=z$C,A=z$A,V=z$V,iterations=z$iter,converged=z$conv_flag))
  }
  # Native public initialization. Centering belongs to the complete public timer.
  xc <- sweep(X,2,colMeans(X));yc <- sweep(Y,2,colMeans(Y))
  z <- rrpack::srrr(yc,xc,nrank=r,
    modstr=list(lamA=n*lam,nlam=1,WA=w),
    control=list(epsilon=1e-8,maxit=1000,inner.eps=1e-9,inner.maxit=10000))
  list(B=coef(z),A=matrix(z$A.path[1,,],p,r),V=matrix(z$V.path[1,,],q,r),iterations=NA,converged=NA)
}
rss <- function(key="VmRSS") {
  text <- readLines("/proc/self/status");as.numeric(gsub("[^0-9]","",text[grepl(paste0("^",key,":"),text)]))*1024
}
for(i in seq_len(nrow(configs))) invisible(fit(configs[i,]))
metadata <- data.frame(case=case,rep=replication,mode=mode,R=as.character(getRversion()),rrpack=as.character(packageVersion("rrpack")),
  cpu=trimws(sub(".*:","",readLines("/proc/cpuinfo")[grepl("^model name",readLines("/proc/cpuinfo"))][1])),
  openblas_threads=Sys.getenv("OPENBLAS_NUM_THREADS"),omp_threads=Sys.getenv("OMP_NUM_THREADS"),mkl_threads=Sys.getenv("MKL_NUM_THREADS"),
  warmed_current_rss=rss(),warmed_peak_rss=rss("VmHWM"),X_md5=unname(tools::md5sum(file.path(input,"X.bin"))),Y_md5=unname(tools::md5sum(file.path(input,"Y.bin"))))
rows <- list();saved <- list()
for(rd in 1:3) for(i in ((seq_len(nrow(configs))+rd+replication-2)%%nrow(configs))+1) {
  c <- configs[i,];key <- tag(c);gc();start <- Sys.time();z <- fit(c);elapsed <- as.numeric(difftime(Sys.time(),start,units="secs"))
  scratch <- file.path(out,"round-B.bin");savebin(z$B,scratch);digest <- unname(tools::md5sum(scratch))
  rows[[length(rows)+1]] <- data.frame(variant="r",case=case,rep=replication,config=key,round=rd,seconds=elapsed,
    iterations=z$iterations,converged=z$converged,B_md5=digest)
  if(rd==1) {
    saved[[key]] <- z
    for(k in c("B","A","V")) savebin(z[[k]],file.path(out,paste0(key,"-",k,".bin")))
    savebin(if(c$adaptive) adaptive else rep(1,p),file.path(out,paste0(key,"-weights.bin")))
  }
  write.csv(do.call(rbind,rows),file.path(out,"timings.csv"),row.names=FALSE)
}
metadata$current_after_timing <- rss();metadata$peak_after_timing <- rss("VmHWM")
if(case %in% c("wide","responses","ill")) for(i in seq_len(nrow(configs))) {
  c <- configs[i,];key <- tag(c);calls <- 0;start <- Sys.time()
  Rprof(file.path(out,paste0("cpu-",key,".Rprof")),interval=.001,memory.profiling=TRUE)
  repeat {invisible(fit(c));calls <- calls+1;if(calls>=20 || as.numeric(difftime(Sys.time(),start,units="secs"))>=.25) break}
  Rprof(NULL)
  capture.output(cat("calls=",calls,"\n"),summaryRprof(file.path(out,paste0("cpu-",key,".Rprof")),memory="both"),file=file.path(out,paste0("cpu-",key,".txt")))
}
metadata$whole_worker_peak_before_replay <- rss("VmHWM")
# Public wrapper omits iteration/flag fields. Capture its unchanged compiled
# solver return only during untimed replay; verify the timed coefficients.
if(mode=="public") trace("srrr_Rcpp",where=asNamespace("rrpack"),
  exit=quote(assign("HDM102_CAPTURE",returnValue(),envir=.GlobalEnv)),print=FALSE)
probes <- list()
for(i in seq_len(nrow(configs))) {
  c <- configs[i,];key <- tag(c);z <- fit(c);ref <- saved[[key]]
  err <- sqrt(sum((z$B-ref$B)^2))/max(1,sqrt(sum(ref$B^2)))
  stopifnot(is.finite(err),err<=1e-12)
  native <- if(mode=="public") HDM102_CAPTURE else list(iter=z$iterations,conv_flag=z$converged)
  probes[[length(probes)+1]] <- data.frame(config=key,iterations=native$iter,converged=native$conv_flag,
    replay_coefficient_error=err,inner_counts="unavailable from installed solver")
}
if(mode=="public") untrace("srrr_Rcpp",where=asNamespace("rrpack"))
write.csv(do.call(rbind,probes),file.path(out,"probes.csv"),row.names=FALSE)
metadata$whole_worker_peak_with_replay <- rss("VmHWM")
write.csv(metadata,file.path(out,"metadata.csv"),row.names=FALSE)
capture.output(sessionInfo(),extSoftVersion(),Sys.getenv(c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),
  file=file.path(out,"environment.txt"))
capture.output(rrpack::srrr,rrpack:::srrr_Rcpp,rrpack::rrr.fit,file=file.path(out,"installed-source.R"))
