# New data for validating the single candidate chosen on seeds8201:8203.
# The previous selected glmgen controls are transported without a new sweep.
args <- commandArgs(TRUE);out <- args[1];prior <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
cases <- read.delim(file.path(prior,"inputs/cases.tsv"))
cases <- cases[cases$family=="signal" & cases$order==1,]
cases$genlasso <- FALSE
cases <- rbind(cases,data.frame(id="million-k1",family="signal",n=1000000L,
  p=1000000L,order=1,grid="unit",path=0,lo=1,hi=1,genlasso=FALSE))
settings <- read.delim(file.path(prior,"final-report/optimized-selected-ratios.tsv"),check.names=FALSE)
keys <- ifelse(cases$id=="million-k1","signal-n100000-k1-l1",cases$id)
cases$r_setting <- settings$reference[match(keys,settings$case)]
stopifnot(all(cases$r_setting %in% c("glmgen","glmgen-1e8","glmgen-default")))
write.table(cases,file.path(out,"cases.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
for(seed in 8301:8303) {
  set.seed(seed);d <- file.path(out,seed);dir.create(d,showWarnings=FALSE)
  for(n in c(1000L,10000L,100000L,1000000L)) {
    t <- seq(0,1,length.out=n);y <- sin(4*pi*t)+.6*(t>.55)+rnorm(n,sd=.2)
    write.table(y,file.path(d,sprintf("signal-%d-y.tsv",n)),sep="\t",row.names=FALSE,col.names=FALSE,quote=FALSE)
    if(n==10000) write.table(cumsum(runif(n,.8,1.2)),file.path(d,"irregular-x.tsv"),sep="\t",row.names=FALSE,col.names=FALSE,quote=FALSE)
  }
}
