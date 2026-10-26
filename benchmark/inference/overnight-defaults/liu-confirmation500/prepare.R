cases=read.csv('../calibration-v2/calibration-cases.csv');caseid='liu-published';ci=match(caseid,cases$case);case=cases[ci,]
stopifnot(case$design=='liu',case$n==1000,case$p==201)
for(rep in 1:500){
 set.seed(1200000000+ci*10000+rep);n=case$n;p=case$p;P=diag(c(1.5,rep(1.2,p-1)));P[1,2:5]=P[2:5,1]=.2
 U=chol(solve(P));y=rbinom(n,1,.5);mu=c(.4,-.25,-.25,rep(0,p-3));X=matrix(rnorm(n*p),n,p)%*%U+outer(y,mu);truth=as.vector(P%*%mu);ts=c(1,(p-4):p)
 stem=file.path('outputs','liu-confirmation',caseid,rep);dir.create(stem,recursive=TRUE,showWarnings=FALSE)
 for(name in c('X','y','truth','ts')){
  file=file.path(stem,paste0('cal-',name,'.csv'));value=get(name)
  if(file.exists(file)){old=as.matrix(read.csv(file,header=FALSE));stopifnot(max(abs(as.vector(old)-as.vector(value)))<1e-12)}else write.table(value,file,sep=',',row.names=FALSE,col.names=FALSE)
 }
}
writeLines('500 prospective Liu n1000,p201 confirmation inputs at disjoint base1200000000','../liu-confirmation500-inputs.txt')
