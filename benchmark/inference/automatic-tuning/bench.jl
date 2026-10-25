using CoordinateDescent, Random, LinearAlgebra, Statistics, DelimitedFiles
BLAS.set_num_threads(1)
engine,label,rep=ARGS
base=joinpath("bench-data",label,rep); out=joinpath("bench",join(ARGS,'-'))
X=readdlm(joinpath(base,"X.csv"),',',Float64); y=vec(readdlm(joinpath(base,"y.csv"),',',Float64))
c=vec(readdlm(joinpath(base,"config.csv"),',',Float64;skipstart=1)); lam,node,mu=c
opt=CDOptions(;opt_tol=1e-9,max_iter=100000)
function fit(A,b)
 if engine=="ours"
   return debiased_lasso(A,b,lam;nodewise_lambda=node,threads=false,options=opt,rng=Xoshiro(77))
 elseif engine=="oursqp10"
   return debiased_lasso(A,b,lam;method=:qp,mu,targets=1:10,threads=false,options=opt,rng=Xoshiro(77))
 else
   return debiasing_matrix(A;method=:qp,mu,threads=false,options=opt,rng=Xoshiro(77))
 end
end
try
 A=randn(Xoshiro(42),80,12);b=randn(Xoshiro(43),80);fit(A,b);GC.gc()
 t=time_ns();r=fit(X,y);elapsed=(time_ns()-t)/1e9
 if engine=="oursqp"
   M=Matrix(r.theta);G=X'X/size(X,1);gaps=diag(M*G*M')-diag(M)+mu*vec(sum(abs,M;dims=2))
   resid=maximum(abs,M*G-I);gap=maximum(abs,gaps)
   writedlm(out*"-theta.csv",M,',')
   status=all(r.converged) && resid<=mu+1e-6 && gap<1e-6 ? "ok" : "invalid"
   open(out*".meta","w") do f
     println(f,"status=",status,"\nfit_seconds=",elapsed,"\nqp_gap=",gap,"\nqp_residual=",resid,"\ntargets=",size(M,1))
   end
 else
   ci=confint(r)
   writedlm(out*"-coef.csv",hcat(r.beta,r.se,ci),',')
   open(out*".meta","w") do f
     println(f,"status=",r.converged ? "ok" : "invalid","\nfit_seconds=",elapsed,"\ntargets=",length(r.beta))
   end
 end
catch e
 open(out*".meta","w") do f;println(f,"status=error\nerror=",sprint(showerror,e));end
end
