# Compare untouched author source with observation-only instrumentation on saved inputs.
caseid=commandArgs(TRUE)[1]
.libPaths(c('/home1/mkolar/juliaPkg/runs/20261025-cd77-comparison-pilot/Rlib','/home1/mkolar/juliaPkg/Rlib-cd-r4.5.3','/home1/mkolar/juliaPkg/Rlib-r4.5.3',.libPaths()))
suppressPackageStartupMessages({library(Matrix);library(glmnet);library(expm);library(flare)})
X=as.matrix(read.csv(file.path('competitor-data',caseid,'1','X.csv'),header=FALSE));y=scan(file.path('competitor-data',caseid,'1','y.csv'),quiet=TRUE)
root=file.path('reports','sslasso-native-audit',caseid);dir.create(root,recursive=TRUE,showWarnings=FALSE)
src=readLines('/home1/mkolar/juliaPkg/runs/20261025-cd77-inference/references/sslasso_code/lasso_inference.r')
for(mode in c('original','observed')) {
 e=new.env();code=src
 if(mode=='observed') {
  code=sub('M[i,] <- beta;','selected_mu[i] <<- mu; M[i,] <- beta;',code,fixed=TRUE)
  code=sub('unbiased.Lasso <- as.numeric','last_M <<- M; last_G <<- sigma.hat; unbiased.Lasso <- as.numeric',code,fixed=TRUE)
 }
 eval(parse(text=code[!grepl('^library',code)]),envir=e);e$selected_mu=rep(NA_real_,ncol(X)+1)
 set.seed(770001)
 tryCatch(withCallingHandlers({
  f=e$SSLasso(X,y,verbose=FALSE);saveRDS(f,file.path(root,paste0(mode,'.rds')));writeLines('ok',file.path(root,paste0(mode,'.status')))
 },error=function(err){writeLines(vapply(sys.calls(),function(x)paste(deparse(x),collapse=' '),''),file.path(root,paste0(mode,'.trace')))}),error=function(err)writeLines(conditionMessage(err),file.path(root,paste0(mode,'.error'))))
}
if(file.exists(file.path(root,'original.rds'))&&file.exists(file.path(root,'observed.rds'))) {
 a=readRDS(file.path(root,'original.rds'));b=readRDS(file.path(root,'observed.rds'))
 writeLines(paste('identical',identical(a,b)),file.path(root,'parity.txt'))
}
