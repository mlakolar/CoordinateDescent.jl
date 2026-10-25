library(quantreg)
library(rqPen)
library(conquer)
X=as.matrix(read.csv('golden/X.csv',header=FALSE)); y=scan('golden/y.csv',quiet=TRUE)
n=nrow(X); p=ncol(X)
check=function(b,tau,lambda) {
 r=as.numeric(y-cbind(1,X)%*%b)
 mean(pmax(tau*r,(tau-1)*r))+lambda*sum(abs(b[-1]))
}
kkt=function(b,tau,lambda,h) {
 r=as.numeric(y-cbind(1,X)%*%b); s=pmax(-1,pmin(1,r/h))
 u=tau-.5+.75*s-.25*s^3
 g=-as.numeric(crossprod(X,u))/n
 max(abs(mean(u)),ifelse(abs(b[-1])>1e-7,abs(g+lambda*sign(b[-1])),pmax(abs(g)-lambda,0)))
}
rows=list()
for(tau in c(.2,.5,.8)) for(lambda in c(.03,.1)) {
 exact=scan(sprintf('golden/exact-%s-%s.csv',tau,lambda),quiet=TRUE)
 smooth=scan(sprintf('golden/smooth-%s-%s.csv',tau,lambda),quiet=TRUE)
 ref=check(exact,tau,lambda)
 # quantreg adds pseudo-observations with median loss, so its supplied lambda
 # contributes lambda/2 times the l1 norm to the SUM of check losses.
 q=quantreg::rq.fit.lasso(cbind(1,X),y,tau=tau,lambda=2*n*lambda,eps=1e-9)$coefficients
 rp=rqPen::rq.pen(X,y,tau=tau,lambda=c(lambda,lambda*.999999),penalty='LASSO',alg='br',scalex=FALSE,lambda.discard=FALSE)
 rb=as.numeric(rp$models[[1]]$coefficients[,1])
 cq=conquer::conquer.reg(X,y,lambda=lambda,tau=tau,kernel='parabolic',h=.2,penalty='lasso',epsilon=1e-9,iteMax=20000)$coeff
 rows[[length(rows)+1]]=data.frame(tau=tau,lambda=lambda,quantreg_objective_error=check(q,tau,lambda)-ref,rqPen_objective_error=check(rb,tau,lambda)-ref,conquer_smooth_coef_error=max(abs(cq-smooth)),conquer_kkt=kkt(cq,tau,lambda,.2),julia_smooth_kkt=kkt(smooth,tau,lambda,.2))
}
z=do.call(rbind,rows);write.csv(z,'golden/R-comparison-v2.csv',row.names=FALSE);print(z)
stopifnot(max(abs(z$quantreg_objective_error))<1e-6,max(abs(z$rqPen_objective_error))<1e-6,max(z$conquer_kkt)<2e-5,max(z$julia_smooth_kkt)<2e-5,max(z$conquer_smooth_coef_error)<2e-4)
writeLines(capture.output(sessionInfo()),'golden/R-session.txt')
cat('R GOLDEN CHECKS PASSED\n')

writeLines(capture.output(quantreg::rq.fit.lasso),'golden/quantreg-rq-fit-lasso-source.txt')
