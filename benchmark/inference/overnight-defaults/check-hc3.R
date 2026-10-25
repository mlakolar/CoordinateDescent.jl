cases=read.csv('../gaussian-hc3-v1/confirmation-cases.csv');cases=cases[cases$family=='gaussian',]
checks=0;largestSE=0;largestP=0
for(caseid in cases$case) {
 stem=file.path('outputs','confirmation-canary',caseid,'1')
 M=as.matrix(read.csv(paste0(stem,'.selected-design.csv'),header=FALSE));v=as.matrix(read.csv(paste0(stem,'.hc3-moments.csv'),header=FALSE));r=read.csv(paste0(stem,'.confirm.csv'))
 h=rowSums(qr.Q(qr(M))^2);stopifnot(max(abs(h-v[,2]))<1e-10)
 phi=v[,-c(1,2),drop=FALSE]*as.vector(v[,1]/(1-h));phi=sweep(phi,2,colMeans(phi))
 se=sqrt(colSums(phi^2))/nrow(M);pv=2*pnorm(-abs(r$estimate/se))
 largestSE=max(largestSE,max(abs(se-r$se)));largestP=max(largestP,max(abs(pv-r$pvalue)))
 stopifnot(max(abs(se-r$se))<1e-10,max(abs(pv-r$pvalue))<1e-8)
 checks=checks+length(se)+1
}
cat('independent_R_HC3_checks=',checks,' maximum_SE_difference=',largestSE,' maximum_pvalue_difference=',largestP,'\n')
