suppressPackageStartupMessages(library(glmgen))
args<-commandArgs(TRUE);label<-args[1];input<-args[2];out<-args[3]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
parts<-strsplit(label,"-",fixed=TRUE)[[1]]
gridphase<-if(length(args)>=4)args[4] else "main"
grid<-read.csv(file.path(input,paste0("cases-",gridphase,".csv")))
m<-grid[grid$case==parts[1]&grid$rep==as.integer(parts[2]),];stopifnot(nrow(m)==1)
readvec<-function(name,count){con<-file(file.path(input,label,paste0(name,".bin")),"rb");on.exit(close(con));readBin(con,"double",count,size=8,endian="little")}
y<-readvec("y",m$n);x<-if(tolower(as.character(m$irregular))=="true")readvec("x",m$n) else seq_len(m$n)
levels<-if(m$path==0)m$lo else exp(seq(log(m$hi),log(m$lo),length.out=m$path))
# A tighter independent reference, outside the Julia timers. No timing rank
# against this stopping setting is claimed.
z<-glmgen::trendfilter(x,y,k=m$order,lambda=levels,thinning=FALSE,
  control=glmgen::trendfilter.control.list(obj_tol=1e-12,max_iter=20000L))
stopifnot(is.matrix(z$beta),nrow(z$beta)==m$n,ncol(z$beta)==length(levels))
con<-file(file.path(out,"coefficients.bin"),"wb");writeBin(as.double(z$beta),con,size=8,endian="little");close(con)
write.csv(data.frame(level=z$lambda,status=z$status,iterations=z$iter),file.path(out,"status.csv"),row.names=FALSE)
write.csv(data.frame(R=R.version.string,glmgen=as.character(packageVersion("glmgen")),
  y_md5=unname(tools::md5sum(file.path(input,label,"y.bin"))),
  x_md5=unname(tools::md5sum(file.path(input,label,"x.bin"))),
  OPENBLAS_NUM_THREADS=Sys.getenv("OPENBLAS_NUM_THREADS"),OMP_NUM_THREADS=Sys.getenv("OMP_NUM_THREADS")),
  file.path(out,"metadata.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"session.txt"))
capture.output(packageDescription("glmgen"),file=file.path(out,"package.txt"))
stopifnot(all(z$status==0),all(is.finite(z$beta)))
