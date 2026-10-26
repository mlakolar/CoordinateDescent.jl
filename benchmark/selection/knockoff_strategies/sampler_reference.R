root<-Sys.getenv('HDM_REFERENCE');base<-dirname(root)
library(knockoff)
stopifnot(as.character(packageVersion('knockoff'))=='0.3.6')
for(name in c('create.gaussian','create.fixed','create.solve_equi')) {
 writeLines(deparse(getFromNamespace(name,'knockoff')),file.path(root,paste0('R-',name,'-source.txt')))
}
for(case in seq_len(16)) {
 out<-file.path(root,paste0('case-',case));dir.create(out,recursive=TRUE,showWarnings=FALSE)
 S<-as.matrix(read.csv(file.path(base,'objective-suite-v1',paste0('case-',case),'covariance.csv'),header=FALSE));p<-nrow(S)
 set.seed(106100+case);Z<-matrix(rnorm(p*p),p,p)
 set.seed(106100+case);K0<-knockoff::create.gaussian(matrix(0,p,p),rep(0,p),S,method='equi')
 set.seed(106100+case);K1<-knockoff::create.gaussian(diag(p),rep(0,p),S,method='equi')
 for(name in c('Z','K0','K1'))write.table(get(name),file.path(out,paste0('R-',name,'.csv')),sep=',',row.names=FALSE,col.names=FALSE)
 write.table(knockoff::create.solve_equi(S),file.path(out,'R-direct-diagonal.csv'),sep=',',row.names=FALSE,col.names=FALSE)
}
writeLines(c(capture.output(sessionInfo()),paste('RNGkind',paste(RNGkind(),collapse=','))),file.path(root,'R-session.txt'))
