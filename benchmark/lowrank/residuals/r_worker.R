suppressPackageStartupMessages(library(rrpack))
args<-commandArgs(TRUE);input<-args[1];out<-args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
dims<-scan(file.path(input,"dims.txt"),quiet=TRUE);n<-dims[1];p<-dims[2];q<-dims[3]
readmat<-function(name,nr,nc){con<-file(file.path(input,paste0(name,".bin")),"rb");on.exit(close(con));matrix(readBin(con,"double",nr*nc,size=8,endian="little"),nr,nc)}
X<-readmat("X",n,p);Y<-readmat("Y",n,q)
bits<-strsplit(basename(input),"-",fixed=TRUE)[[1]];case<-bits[1];replication<-as.integer(bits[2])
prepare<-function(){list(X=sweep(X,2,colMeans(X)),Y=sweep(Y,2,colMeans(Y)))}
call_rrr<-function(d,ranks){rrpack::rrr(d$Y,d$X,penaltySVD="rank",df.type="naive",maxrank=3,
    modstr=list(lambda=ranks),control=list(qr.tol=max(n,p)*.Machine$double.eps,sv.tol=1e-12))}
fit_one<-function(){d<-prepare();z<-call_rrr(d,3);list(B=coef(z),native=z)}
fit_path<-function(){
  d<-prepare();z<-call_rrr(d,1:3)
  # coef(z) contains only the IC-selected fit. Recover the stored native path
  # from the documented coef.ls, A and Spath, sharing its QR/SVD preparation.
  coefs<-lapply(1:3,function(i){
    r<-sum(z$Spath[,i]!=0);A<-z$A[,seq_len(r),drop=FALSE]
    (z$coef.ls %*% sweep(A,2,z$Spath[seq_len(r),i],"*")) %*% t(A)
  })
  list(B=c(list(matrix(0,p,q)),coefs),native=z)
}
writebin<-function(value,name){con<-file(file.path(out,name),"wb");writeBin(as.double(value),con,size=8,endian="little");close(con)}
persist<-function(z,mode){
  if(mode=="public_fit")writebin(z$B,"fit-3-B.bin")
  else for(r in 0:3)writebin(z$B[[r+1]],paste0("path-",r,"-B.bin"))
}
rss<-function(key){s<-readLines("/proc/self/status");as.numeric(gsub("[^0-9]","",s[grepl(paste0("^",key,":"),s)]))*1024}
functions<-list(public_fit=fit_one,public_path=fit_path)
for(fn in functions)invisible(fn())
metadata<-data.frame(variant="r",case=case,rep=replication,R=R.version.string,rrpack=as.character(packageVersion("rrpack")),
  X_md5=unname(tools::md5sum(file.path(input,"X.bin"))),Y_md5=unname(tools::md5sum(file.path(input,"Y.bin"))),
  warmed_current_rss=rss("VmRSS"),warmed_peak_rss=rss("VmHWM"))
con<-file(file.path(out,"timings.csv"),"w")
writeLines("variant,case,rep,mode,round,seconds,bytes,gctime",con)
for(rd in 1:3){
  modes<-if((replication+rd)%%2==0)names(functions) else rev(names(functions))
  for(mode in modes){
    invisible(gc());began<-as.numeric(Sys.time());value<-functions[[mode]]();elapsed<-as.numeric(Sys.time())-began
    writeLines(paste("r",case,replication,mode,rd,format(elapsed,digits=17),"","",sep=","),con);flush(con)
    if(rd==1)persist(value,mode)
  }
}
close(con)
metadata$peak_after_timing<-rss("VmHWM");metadata$current_after_timing<-rss("VmRSS")
if(case %in% c("responses","wide")){
  for(mode in names(functions)){
    path<-file.path(out,paste0("cpu-",mode,".Rprof"));Rprof(path,interval=.001)
    began<-as.numeric(Sys.time());calls<-0
    repeat{functions[[mode]]();calls<-calls+1;if(calls>=20 || as.numeric(Sys.time())-began>.25)break}
    Rprof(NULL)
    capture.output(summaryRprof(path),file=file.path(out,paste0("cpu-",mode,".txt")))
  }
}
metadata$whole_worker_peak_rss<-rss("VmHWM")
write.csv(metadata,file.path(out,"metadata.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"session.txt"))
