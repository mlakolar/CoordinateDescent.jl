args <- commandArgs(TRUE); root <- args[1]; output <- args[2]
d <- read.delim(file.path(output,"optimized-selected-ratios.tsv"),check.names=FALSE)
cases <- read.delim(file.path(root,"inputs/cases.tsv"))
d <- merge(d,cases[,c("id","lo","grid")],by.x="case",by.y="id",sort=FALSE)
ok <- tolower(d$accuracy_pass)=="true" & tolower(d$coverage_and_convergence_pass)=="true"
dir.create(output,recursive=TRUE,showWarnings=FALSE)
colors <- c("0.1"="#2463a8","1"="#d27516","10"="#27805c")
png_type <- if(Sys.info()[["sysname"]]=="Darwin") "quartz" else "cairo"
png(file.path(output,"signal-comparison.png"),width=1800,height=650,res=160,type=png_type)
par(mfrow=c(1,3),mar=c(4.5,4.5,3,1),oma=c(1.5,0,1.5,0),cex=.9)
signals <- d[d$family=="signal" & d$path==0 & ok,]
ylim <- range(c(signals$min_ratio,signals$max_ratio,1))
ylim <- 10^c(floor(log10(ylim[1])),ceiling(log10(ylim[2])))
for(k in 0:2) {
  z <- signals[signals$order==k,]
  plot(NA,xlim=c(800,130000),ylim=ylim,log="xy",xaxt="n",xlab="Signal length",ylab="R time / Julia time",
    main=paste("Order",k))
  axis(1,at=c(1000,10000,100000),labels=c("1,000","10,000","100,000"))
  abline(h=1,col="#555555",lwd=1.4,lty=2)
  for(lam in c(.1,1,10)) {
    a <- z[z$lo==lam & z$grid=="unit",]; a <- a[order(a$n),]
    color <- colors[as.character(lam)]
    lines(a$n,a$reference_over_julia,col=color,lwd=1.5)
    points(a$n,a$reference_over_julia,col=color,pch=16)
    arrows(a$n,a$min_ratio,a$n,a$max_ratio,col=color,length=.03,angle=90,code=3)
    b <- z[z$lo==lam & z$grid=="irregular",]
    points(b$n*1.08,b$reference_over_julia,col=color,pch=17)
  }
  if(k==0) legend("bottomleft",legend=c("R lambda = 0.1","R lambda = 1","R lambda = 10","Irregular grid"),
    col=c(colors,"#333333"),pch=c(16,16,16,17),bty="n",cex=.8)
}
mtext("Fastest tested settings passing coefficient and objective agreement on all three datasets",outer=TRUE,side=3,line=.1,cex=.85)
mtext("Above 1: Julia faster. Bars: range of paired dataset ratios; three timing rounds per dataset. Failed cases omitted.",outer=TRUE,side=1,line=.1,cex=.75)
dev.off()

png(file.path(output,"regression-comparison.png"),width=1500,height=950,res=160,type=png_type)
par(mar=c(6,10,3,1),cex=.9)
reg <- d[d$family!="signal",]
key <- function(a) paste(a$family,ifelse(a$path>0,"50-level path",paste0("lambda = ",a$lo/a$n)))
reg$key <- key(reg)
labels <- unique(reg$key)
xlim <- range(c(reg$min_ratio[ok[d$family!="signal"]],reg$max_ratio[ok[d$family!="signal"]],1))
xlim <- 10^c(floor(log10(xlim[1])),ceiling(log10(xlim[2])))
plot(NA,xlim=xlim,ylim=c(.5,length(labels)+.5),log="x",yaxt="n",ylab="",xlab="genlasso time / Julia time",main="Regression fits and tuning paths")
axis(2,at=seq_along(labels),labels=labels,las=1,cex.axis=.85)
abline(v=1,col="#555555",lty=2,lwd=1.4)
for(p in c(40,100)) {
  z <- reg[reg$p==p,]
  pos <- match(z$key,labels)+if(p==40) -.12 else .12
  color <- if(p==40) "#2463a8" else "#d27516"
  pass <- tolower(z$accuracy_pass)=="true" & tolower(z$coverage_and_convergence_pass)=="true"
  arrows(z$min_ratio[pass],pos[pass],z$max_ratio[pass],pos[pass],col=color,length=.03,angle=90,code=3)
  points(z$reference_over_julia[pass],pos[pass],col=color,pch=16)
  points(rep(1,sum(!pass)),pos[!pass],col=color,pch=4)
}
legend("topright",legend=c("n=200, p=40","n=1,000, p=100","Accuracy or convergence failure"),
  col=c("#2463a8","#d27516","#555555"),pch=c(16,16,4),bty="n",cex=.8)
mtext("Right of 1: Julia faster. Fastest passing genlasso routine; identical objectives and requested grids.",side=1,line=4.5,cex=.75)
dev.off()
