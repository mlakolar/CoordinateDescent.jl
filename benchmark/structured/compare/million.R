# Separate inputs: never regenerate or change the active main campaign inputs.
out <- commandArgs(TRUE)[1]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
n <- 1000000L
cases <- data.frame(id=paste0("million-k",0:2),family="signal",n=n,p=n,order=0:2,
  grid="unit",path=0L,lo=1,hi=1,genlasso=FALSE)
calibration <- read.delim("final-analysis/optimized-selected-ratios.tsv",check.names=FALSE)
settings <- calibration[match(paste0("signal-n100000-k",0:2,"-l1"),calibration$case),]
stopifnot(all(tolower(settings$accuracy_pass)=="true"),
          all(tolower(settings$coverage_and_convergence_pass)=="true"))
cases$r_setting <- settings$reference
cases$julia_setting <- settings$julia_setting
write.table(cases,file.path(out,"cases.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
for(seed in 8201:8203) {
  set.seed(seed+10000L)
  t <- seq(0,1,length.out=n); y <- sin(4*pi*t)+.6*(t>.55)+rnorm(n,sd=.2)
  dir.create(file.path(out,seed),showWarnings=FALSE)
  write.table(y,file.path(out,seed,paste0("signal-",n,"-y.tsv")),sep="\t",row.names=FALSE,col.names=FALSE,quote=FALSE)
}
